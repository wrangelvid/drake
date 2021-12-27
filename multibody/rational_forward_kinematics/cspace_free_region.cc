#include "drake/multibody/rational_forward_kinematics/cspace_free_region.h"

#include <fmt/format.h>

#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics_internal.h"

namespace drake {
namespace multibody {

namespace {
struct DirectedKinematicsChain {
  DirectedKinematicsChain(BodyIndex m_start, BodyIndex m_end)
      : start(m_start), end(m_end) {}

  bool operator==(const DirectedKinematicsChain& other) const {
    return start == other.start && end == other.end;
  }

  BodyIndex start;
  BodyIndex end;
};

struct DirectedKinematicsChainHash {
  size_t operator()(const DirectedKinematicsChain& p) const {
    return p.start * 100 + p.end;
  }
};
}  // namespace

CspaceFreeRegion::CspaceFreeRegion(
    const MultibodyPlant<double>& plant,
    const std::vector<const ConvexPolytope*>& link_polytopes,
    const std::vector<const ConvexPolytope*>& obstacles,
    SeparatingPlaneOrder plane_order, CspaceRegionType cspace_region_type)
    : rational_forward_kinematics_(plant),
      obstacles_{obstacles},
      plane_order_{plane_order},
      cspace_region_type_{cspace_region_type} {
  // First group the link polytopes by the attached link.
  for (const auto& link_polytope : link_polytopes) {
    DRAKE_DEMAND(link_polytope->body_index() != plant.world_body().index());
    const auto it = link_polytopes_.find(link_polytope->body_index());
    if (it == link_polytopes_.end()) {
      link_polytopes_.emplace_hint(
          it,
          std::make_pair(link_polytope->body_index(),
                         std::vector<const ConvexPolytope*>({link_polytope})));
    } else {
      it->second.push_back(link_polytope);
    }
  }
  // Now create the separating planes.
  // By default, we only consider the pairs between a link polytope and a
  // world obstacle.
  // TODO(hongkai.dai): consider the self collision between link obstacles.
  separating_planes_.reserve(link_polytopes.size() * obstacles.size());
  // If we verify an axis-algined bounding box in the C-space, then for each
  // pair of (link, obstacle), then we only need to consider variables t for the
  // joints on the kinematics chain between the link and the obstacle; if we
  // verify a generic bounding box, then we need to consider t for all joint
  // angles.
  std::unordered_map<SortedPair<BodyIndex>, VectorX<symbolic::Variable>>
      map_link_obstacle_to_t;
  for (const auto& obstacle : obstacles_) {
    DRAKE_DEMAND(obstacle->body_index() == plant.world_body().index());
    for (const auto& [link, polytopes_on_link] : link_polytopes_) {
      for (const auto& link_polytope : polytopes_on_link) {
        Vector3<symbolic::Expression> a;
        symbolic::Expression b;
        const symbolic::Monomial monomial_one{};
        VectorX<symbolic::Variable> plane_decision_vars;
        if (plane_order_ == SeparatingPlaneOrder::kConstant) {
          plane_decision_vars.resize(4);
          for (int i = 0; i < 3; ++i) {
            plane_decision_vars(i) = symbolic::Variable(
                "a" + std::to_string(separating_planes_.size() * 3 + i));
            plane_decision_vars(3) = symbolic::Variable(
                "b" + std::to_string(separating_planes_.size()));
          }
          a = plane_decision_vars.head<3>().cast<symbolic::Expression>();
          b = plane_decision_vars(3);
        } else if (plane_order_ == SeparatingPlaneOrder::kAffine) {
          VectorX<symbolic::Variable> t_for_plane;
          if (cspace_region_type_ == CspaceRegionType::kGenericPolytope) {
            t_for_plane = rational_forward_kinematics_.t();
          } else if (cspace_region_type_ ==
                     CspaceRegionType::kAxisAlignedBoundingBox) {
            SortedPair<BodyIndex> link_obstacle(link_polytope->body_index(),
                                                obstacle->body_index());
            auto it = map_link_obstacle_to_t.find(link_obstacle);
            if (it == map_link_obstacle_to_t.end()) {
              t_for_plane = rational_forward_kinematics_.FindTOnPath(
                  obstacle->body_index(), link_polytope->body_index());
              map_link_obstacle_to_t.emplace_hint(it, link_obstacle,
                                                  t_for_plane);
            } else {
              t_for_plane = it->second;
            }
          }
          // Now create the variable a_coeff, a_constant, b_coeff, b_constant,
          // such that a = a_coeff * t_for_plane + a_constant, and b = b_coeff *
          // t_for_plane + b_constant.
          Matrix3X<symbolic::Variable> a_coeff(3, t_for_plane.rows());
          Vector3<symbolic::Variable> a_constant;
          for (int i = 0; i < 3; ++i) {
            a_constant(i) = symbolic::Variable(
                fmt::format("a_constant{}({})", separating_planes_.size(), i));
            for (int j = 0; j < t_for_plane.rows(); ++j) {
              a_coeff(i, j) = symbolic::Variable(fmt::format(
                  "a_coeff{}({}, {})", separating_planes_.size(), i, j));
            }
          }
          VectorX<symbolic::Variable> b_coeff(t_for_plane.rows());
          for (int i = 0; i < t_for_plane.rows(); ++i) {
            b_coeff(i) = symbolic::Variable(
                fmt::format("b_coeff{}({})", separating_planes_.size(), i));
          }
          symbolic::Variable b_constant(
              fmt::format("b_constant{}", separating_planes_.size()));
          // Now construct a(i) = a_coeff.row(i) * t_for_plane + a_constant(i).
          a = a_coeff * t_for_plane + a_constant;
          b = b_coeff.cast<symbolic::Expression>().dot(t_for_plane) +
              b_constant;
          // Now put a_coeff, a_constant, b_coeff, b_constant to
          // plane_decision_vars.
          plane_decision_vars.resize(4 * t_for_plane.rows() + 4);
          int var_count = 0;
          for (int i = 0; i < 3; ++i) {
            plane_decision_vars.segment(var_count, t_for_plane.rows()) =
                a_coeff.row(i);
            var_count += t_for_plane.rows();
          }
          plane_decision_vars.segment<3>(var_count) = a_constant;
          var_count += 3;
          plane_decision_vars.segment(var_count, t_for_plane.rows()) = b_coeff;
          var_count += t_for_plane.rows();
          plane_decision_vars(var_count) = b_constant;
          var_count++;
        }
        separating_planes_.emplace_back(
            a, b, link_polytope, obstacle,
            internal::FindBodyInTheMiddleOfChain(plant, obstacle->body_index(),
                                                 link_polytope->body_index()),
            plane_order_, plane_decision_vars);
        map_polytopes_to_separating_planes_.emplace(
            SortedPair<ConvexGeometry::Id>(link_polytope->get_id(),
                                           obstacle->get_id()),
            &(separating_planes_[separating_planes_.size() - 1]));
      }
    }
  }
}

std::vector<LinkVertexOnPlaneSideRational>
CspaceFreeRegion::GenerateLinkOnOneSideOfPlaneRationals(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs)
    const {
  std::unordered_map<DirectedKinematicsChain,
                     RationalForwardKinematics::Pose<symbolic::Polynomial>,
                     DirectedKinematicsChainHash>
      body_pair_to_X_AB_multilinear;
  std::vector<LinkVertexOnPlaneSideRational> rationals;
  for (const auto& separating_plane : separating_planes_) {
    if (!IsGeometryPairCollisionIgnored(
            separating_plane.positive_side_polytope->get_id(),
            separating_plane.negative_side_polytope->get_id(),
            filtered_collision_pairs)) {
      // First compute X_AB for both side of the polytopes.
      for (const PlaneSide plane_side :
           {PlaneSide::kPositive, PlaneSide::kNegative}) {
        const ConvexPolytope* polytope;
        const ConvexPolytope* other_side_polytope;
        if (plane_side == PlaneSide::kPositive) {
          polytope = separating_plane.positive_side_polytope;
          other_side_polytope = separating_plane.negative_side_polytope;
        } else {
          polytope = separating_plane.negative_side_polytope;
          other_side_polytope = separating_plane.positive_side_polytope;
        }
        const DirectedKinematicsChain expressed_to_link(
            separating_plane.expressed_link, polytope->body_index());
        auto it = body_pair_to_X_AB_multilinear.find(expressed_to_link);
        if (it == body_pair_to_X_AB_multilinear.end()) {
          body_pair_to_X_AB_multilinear.emplace_hint(
              it, expressed_to_link,
              rational_forward_kinematics_.CalcLinkPoseAsMultilinearPolynomial(
                  q_star, polytope->body_index(),
                  separating_plane.expressed_link));
        }
        it = body_pair_to_X_AB_multilinear.find(expressed_to_link);
        const RationalForwardKinematics::Pose<symbolic::Polynomial>&
            X_AB_multilinear = it->second;
        const std::vector<LinkVertexOnPlaneSideRational>
            rationals_expressed_to_link =
                GenerateLinkOnOneSideOfPlaneRationalFunction(
                    rational_forward_kinematics_, polytope, other_side_polytope,
                    X_AB_multilinear, separating_plane.a, separating_plane.b,
                    plane_side, separating_plane.order);
        // I cannot use "insert" function to append vectors, since
        // LinkVertexOnPlaneSideRational contains const members, hence it does
        // not have an assignment operator.
        std::copy(rationals_expressed_to_link.begin(),
                  rationals_expressed_to_link.end(),
                  std::back_inserter(rationals));
      }
    }
  }
  return rationals;
}

std::vector<LinkVertexOnPlaneSideRational>
GenerateLinkOnOneSideOfPlaneRationalFunction(
    const RationalForwardKinematics& rational_forward_kinematics,
    const ConvexPolytope* polytope, const ConvexPolytope* other_side_polytope,
    const RationalForwardKinematics::Pose<symbolic::Polynomial>&
        X_AB_multilinear,
    const drake::Vector3<symbolic::Expression>& a_A,
    const symbolic::Expression& b, PlaneSide plane_side,
    SeparatingPlaneOrder plane_order) {
  std::vector<LinkVertexOnPlaneSideRational> rational_fun;
  rational_fun.reserve(polytope->p_BV().cols());
  const symbolic::Monomial monomial_one{};
  // a_A and b are not polynomial of sinθ or cosθ.
  Vector3<symbolic::Polynomial> a_A_poly;
  for (int i = 0; i < 3; ++i) {
    a_A_poly(i) = symbolic::Polynomial({{monomial_one, a_A(i)}});
  }
  const symbolic::Polynomial b_poly({{monomial_one, b}});
  for (int i = 0; i < polytope->p_BV().cols(); ++i) {
    // Step 1: Compute vertex position.
    const Vector3<drake::symbolic::Polynomial> p_AVi =
        X_AB_multilinear.p_AB + X_AB_multilinear.R_AB * polytope->p_BV().col(i);

    // Step 2: Compute a_A.dot(p_AVi) + b
    const drake::symbolic::Polynomial point_on_hyperplane_side =
        a_A_poly.dot(p_AVi) + b_poly;

    // Step 3: Convert the multilinear polynomial to rational function.
    rational_fun.emplace_back(
        rational_forward_kinematics
            .ConvertMultilinearPolynomialToRationalFunction(
                plane_side == PlaneSide::kPositive
                    ? point_on_hyperplane_side - 1
                    : -1 - point_on_hyperplane_side),
        polytope, X_AB_multilinear.frame_A_index, other_side_polytope, a_A, b,
        plane_side, plane_order);
  }
  return rational_fun;
}

bool IsGeometryPairCollisionIgnored(
    const SortedPair<ConvexGeometry::Id>& geometry_pair,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs) {
  return filtered_collision_pairs.count(geometry_pair) > 0;
}

bool IsGeometryPairCollisionIgnored(
    ConvexGeometry::Id id1, ConvexGeometry::Id id2,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs) {
  return IsGeometryPairCollisionIgnored(
      drake::SortedPair<ConvexGeometry::Id>(id1, id2),
      filtered_collision_pairs);
}

}  // namespace multibody
}  // namespace drake
