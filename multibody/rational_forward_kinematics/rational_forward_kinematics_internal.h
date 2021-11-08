#pragma once

// Note: the user should not include this header in their code. This header is
// created for internal test only.

#include <unordered_set>

#include "drake/multibody/plant/multibody_plant.h"

namespace drake {
namespace multibody {
namespace internal {
/**
 * In the reshuffled tree (where link A is treated as the root of the tree,
 * instead of the world link), we need to re-compute the parent-child
 * relationship. Hence ReshuffledBody stores the reshuffled parent, children,
 * and the mobilizer between the body and the reshuffled parent.
 */
struct ReshuffledBody {
  ReshuffledBody(BodyIndex m_body_index, const ReshuffledBody* const m_parent,
                 const Mobilizer<double>* const m_mobilizer)
      : body_index{m_body_index}, parent{m_parent}, mobilizer{m_mobilizer} {}
  // The index of this body in the original tree.
  BodyIndex body_index;
  // The parent of this body in the reshuffled tree.
  const ReshuffledBody* const parent;
  // The children of this body in the reshuffled tree.
  std::vector<std::unique_ptr<ReshuffledBody>> children;
  // The mobilizer between this body and the parent in the reshuffled tree.
  const Mobilizer<double>* const mobilizer;
};

/**
 * Given a multibody plant, rebuild its kinematics tree at a specified root.
 */
void ReshuffleKinematicsTree(const MultibodyPlant<double>& plant,
                             ReshuffledBody* root);

/**
 * Find and add all the children to the reshuffled body in the reshuffled tree.
 * @param visited Keeps the indices of the body in the reshuffled tree that has
 * been visited, in the process of building the reshuffled tree.
 */
void AddChildrenToReshuffledBody(const MultibodyPlant<double>& plant,
                                 ReshuffledBody* body,
                                 std::unordered_set<BodyIndex>* visited);

/**
 * Find the shortest path on the kinematics tree from start to the end.
 */
std::vector<BodyIndex> FindShortestPath(const MultibodyPlant<double>& plant,
                                        BodyIndex start, BodyIndex end);

std::vector<MobilizerIndex> FindMobilizersOnShortestPath(
    const MultibodyPlant<double>& plant, BodyIndex start, BodyIndex end);

/**
 * Find the body in the middle of the kinematics chain that goes from the start
 * to the end. Notice that we ignore the welded joint, and only count revolute
 * joint as one step along the chain.
 */
BodyIndex FindBodyInTheMiddleOfChain(const MultibodyPlant<double>& plant,
                                     BodyIndex start, BodyIndex end);
}  // namespace internal
}  // namespace multibody
}  // namespace drake
