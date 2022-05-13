#include "drake/multibody/plant/slender_body.h"

#include "drake/math/differentiable_norm.h"

namespace drake {
namespace multibody {

using math::RigidTransform;

template <typename T>
SlenderBody<T>::SlenderBody(BodyIndex body_index, double transverse_surface_area,
              double longitudinal_surface_area, double transverse_CD, double longitudinal_CD,
              const math::RigidTransform<double>& X_BodyCP,
              double fluid_density)
    : systems::LeafSystem<T>(systems::SystemTypeTag<SlenderBody>{}),
      body_index_(body_index),
      X_BodyCP_(X_BodyCP),
      transverse_surface_area_(transverse_surface_area),
      longitudinal_surface_area_(longitudinal_surface_area),
      transverse_CD_(transverse_CD),
      longitudinal_CD_(longitudinal_CD),
      default_fluid_density_(fluid_density) {
  const systems::InputPortIndex body_poses_index =
      this->DeclareAbstractInputPort("body_poses",
                                     Value<std::vector<RigidTransform<T>>>())
          .get_index();

  this->DeclareAbstractInputPort("body_spatial_velocities",
                                 Value<std::vector<SpatialVelocity<T>>>());

  this->DeclareVectorInputPort("wind_velocity_at_aerodynamic_center", 3);

  this->DeclareVectorInputPort("fluid_density", 1);

  this->DeclareAbstractOutputPort("spatial_force", &SlenderBody<T>::CalcSpatialForce);

  this->DeclareVectorOutputPort("aerodynamic_center", 3,
                                &SlenderBody<T>::CalcAerodynamicCenter,
                                {this->input_port_ticket(body_poses_index)});
}

template <typename T>
SlenderBody<T>* SlenderBody<T>::AddToBuilder(systems::DiagramBuilder<T>* builder,
                               const multibody::MultibodyPlant<T>* plant,
                               const BodyIndex& body_index, double transverse_surface_area,
                               double longitudinal_surface_area, double transverse_CD,
                               double longitudinal_CD,
                               const math::RigidTransform<double>& X_BodyCP,
                               double fluid_density) {
  SlenderBody<T>* slenderbody = builder->template AddSystem<SlenderBody<T>>(
      body_index, transverse_surface_area, longitudinal_surface_area, transverse_CD,
      longitudinal_CD, X_BodyCP, fluid_density);
  builder->Connect(plant->get_body_poses_output_port(),
                   slenderbody->get_body_poses_input_port());
  builder->Connect(plant->get_body_spatial_velocities_output_port(),
                   slenderbody->get_body_spatial_velocities_input_port());
  builder->Connect(slenderbody->get_spatial_force_output_port(),
                   plant->get_applied_spatial_force_input_port());
  return slenderbody;
}

template <typename T>
void SlenderBody<T>::CalcSpatialForce(
    const systems::Context<T>& context,
    std::vector<ExternallyAppliedSpatialForce<T>>* spatial_force) const {
  spatial_force->resize(1);
  const RigidTransform<T>& X_WorldBody =
      get_body_poses_input_port().template Eval<std::vector<RigidTransform<T>>>(
          context)[body_index_];
  const SpatialVelocity<T>& V_WorldBody =
      get_body_spatial_velocities_input_port()
          .template Eval<std::vector<SpatialVelocity<T>>>(context)[body_index_];

  const T fluid_density = get_fluid_density_input_port().HasValue(context)
                              ? get_fluid_density_input_port().Eval(context)[0]
                              : default_fluid_density_;
  Vector3<T> v_WorldWind = Vector3<T>::Zero();
  if (get_wind_velocity_input_port().HasValue(context)) {
    v_WorldWind = get_wind_velocity_input_port().Eval(context);
  }
  const Vector3<T> v_WindBody_World =
      -v_WorldWind + V_WorldBody.translational();

  const math::RotationMatrix<T> R_WorldCP =
      X_WorldBody.rotation() * X_BodyCP_.rotation().template cast<T>();

  const Vector3<T> v_WindBody_CP = R_WorldCP.transpose() * v_WindBody_World;

  SpatialForce<T> F_CP_CP = SpatialForce<T>::Zero();

  const T velocity_norm_x = math::DifferentiableNorm(Vector1<T>(v_WindBody_CP[0]));
  const T velocity_norm_y = math::DifferentiableNorm(Vector1<T>(v_WindBody_CP[1]));
  const T velocity_norm_z = math::DifferentiableNorm(Vector1<T>(v_WindBody_CP[2]));

  F_CP_CP.translational()[0] = - fluid_density * transverse_surface_area_ * transverse_CD_
                                   * 0.5 * v_WindBody_CP[0] * velocity_norm_x;

  F_CP_CP.translational()[1] = - fluid_density * longitudinal_surface_area_ * longitudinal_CD_
                                   * 0.5 * v_WindBody_CP[1] * velocity_norm_y;

  F_CP_CP.translational()[2] = - fluid_density * longitudinal_surface_area_ * longitudinal_CD_ 
                                   * 0.5 * v_WindBody_CP[2] * velocity_norm_z;



  ExternallyAppliedSpatialForce<T>& force = spatial_force->at(0);
  force.body_index = body_index_;
  force.p_BoBq_B = X_BodyCP_.translation();
  force.F_Bq_W = R_WorldCP * F_CP_CP;
}

template <typename T>
void SlenderBody<T>::CalcAerodynamicCenter(
    const systems::Context<T>& context,
    systems::BasicVector<T>* aerodynamic_center) const {
  const RigidTransform<T>& X_WorldBody =
      get_body_poses_input_port().template Eval<std::vector<RigidTransform<T>>>(
          context)[body_index_];

  aerodynamic_center->SetFromVector(
      X_WorldBody * X_BodyCP_.translation().template cast<T>());
}

}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class drake::multibody::SlenderBody)
