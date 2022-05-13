#pragma once

#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/plant/externally_applied_spatial_force.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/body.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace multibody {

/** A System that connects to the MultibodyPlant in order to model the
simplified dynamics a Slender Body. The Slender-Body Theory is described
by a local radius R(x) that varies from 0 <= x <= L. The Area at x = 0 is
assumed to be zero, while the rear area can be anything.
The classical model supports invisicid flow s.t. it stays attached over the body.
Further the flow along the transverse direction is assumed to be quasi static
about the body crosssection.
Thus the model may loose its validy for hypersonic Mach numbers.
a particular model of flat-plate aerodynamics with

as documented in
Flight Vehicle Aerodyanmics by Mark Drela

@system
name: SlenderBody
input_ports:
- body_poses
- body_spatial_velocities
- wind_velocity_at_aerodynamic_center (optional)
- fluid_density (optional)
output_ports:
- spatial_force
- aerodynamic_center
@endsystem

- The optional wind velocity input is a three-element BasicVector<T>
  representing the translational velocity of the wind in world coordinates at
  the aerodynamic center relative to the world origin.  See
  get_aerodynamic_center_output_port() for more details.
- It is expected that the body_poses input should be connected to the
  @ref MultibodyPlant::get_body_poses_output_port() "MultibodyPlant body_poses
  output port" and that body_spatial_velocities input should be connected to
  the @ref MultibodyPlant::get_body_spatial_velocities_output_port()
  "MultibodyPlant body_spatial_velocities output port"
- The output is of type std::vector<ExternallyAppliedSpatialForce<T>>; it is
  expected that this output will be connected to the @ref
  MultibodyPlant::get_applied_spatial_force_input_port()
  "externally_applied_spatial_force input port" of the MultibodyPlant.

@tparam_default_scalar
@ingroup multibody_systems
*/
template <typename T>
class SlenderBody final : public systems::LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SlenderBody);

  /// The density of dry air at 20 deg Celsius at sea-level.
  static constexpr double kDefaultFluidDensity{1.204};

  /** Constructs a system describing a single slender body using slender body theroy
   aerodynamics as described in the class documentation.
   @param body_index indicates the body on which the aerodynamic forces are
   applied.
   @param transverse_surface_area is the total surface area of the transverse
   crosssection in meters squared
   @param longitudinal_surface_area is the total surface area of the longitudinal
   crosssection in meters squared
   @param transverse_CD is the drag coefficient in the transverse direction
   @param longitudinal_CD is the drag coefficient in the longitudinal direction
   @param X_BodyCP is the pose of the slender body frame relative to the body frame,
   whose origin is the aerodynamic cneter of the slender body, the positive x-axis
   points along the chord towards the heading of the body. The y and z-axis can be
   in arbitrary direction since the body is assumed to be symmetric about the x-axis. 
   @param fluid_density is the density of the fluid in kg/m^3. The default
   value is the density of dry air at 20 deg Celsius at sea-level. This value
   is only used if the optional fluid_density input port is not connected.
  */
  SlenderBody(BodyIndex body_index, double transverse_surface_area,
       double longitudinal_surface_area, double transverse_CD,
       double longitudinal_CD,
       const math::RigidTransform<double>& X_BodyCP =
           math::RigidTransform<double>::Identity(),
       double fluid_density = kDefaultFluidDensity);

  /** Scalar-converting copy constructor.  See @ref system_scalar_conversion. */
  template <typename U>
  explicit SlenderBody(const SlenderBody<U>& other)
      : SlenderBody<T>(other.body_index_, other.transverse_surface_area_, other.longitudinal_surface_area_,
                other.transverse_CD_, other.longitudinal_CD_, other.X_BodyCP_,
                other.default_fluid_density_) {}

  /** Returns a reference to the body_poses input port.  It is anticipated
  that this port will be connected the body_poses output port of a
  MultibodyPlant. */
  const systems::InputPort<T>& get_body_poses_input_port() const {
    return this->get_input_port(0);
  }

  /** Returns a reference to the body_spatial_velocities input port.  It is
  anticipated that this port will be connected the body_spatial_velocities
  output port of a MultibodyPlant. */
  const systems::InputPort<T>& get_body_spatial_velocities_input_port() const {
    return this->get_input_port(1);
  }

  /** Returns a reference to the input port for the optional three-element
  BasicVector<T> representing the translational velocity of the wind in world
  coordinates at the aerodynamic center relative to the world origin. If this
  port is not connected, then the wind velocity is taken to be zero. */
  const systems::InputPort<T>& get_wind_velocity_input_port() const {
    return this->get_input_port(2);
  }

  /** Returns a reference to the optional fluid_density input port, which
   accepts a scalar vector in units kg/m^3. This port is provided to support
   vehicles which must take into account variations in atmospheric density;
   such as a spacecraft during re-entry.  If left unconnected, the aerodynamic
   forces will be calculated using the default fluid density passed in the
   constructor. */
  const systems::InputPort<T>& get_fluid_density_input_port() const {
    return this->get_input_port(3);
  }

  /** Returns a reference to the spatial_forces output port.  It is anticipated
  that this port will be connected to the @ref
  MultibodyPlant::get_applied_spatial_force_input_port() "applied_spatial_force"
  input port of a MultibodyPlant. */
  const systems::OutputPort<T>& get_spatial_force_output_port() const {
    return this->get_output_port(0);
  }

  /** Returns a 3-element position of the aerodynamic center of the slender body in
   world coordinates. This output port does not depend on the optional wind
   velocity input port, so it may be used to compute the wind velocity at the
   aerodynamic center without causing any algebraic loops in the Diagram. For
   instance, the following (sub-)Diagram could be used to implement a wind
   field:
                             ┌────────────┐
                      ┌──────┤ Wind Field │◄────┐
                      │      └────────────┘     │
                      │   ┌─────────────────┐   │
                      └──►│   Slender Body  ├───┘
         wind_velocity_at_└─────────────────┘aerodynamic_center
         aerodynamic_center
   */
  const systems::OutputPort<T>& get_aerodynamic_center_output_port() const {
    return this->get_output_port(1);
  }

  /** Helper method that constructs a SlenderBody and connects the input and output
   ports to the MultibodyPlant.

   @param builder is a DiagramBuilder that the SlenderBody will be added to.
   @param plant is the MultibodyPlant containing the body referenced by
   `body_index`, which the SlenderBody ports will be connected to.

   See the SlenderBody constructor for details on the remaining parameters.
   */
  static SlenderBody<T>* AddToBuilder(systems::DiagramBuilder<T>* builder,
                               const multibody::MultibodyPlant<T>* plant, const BodyIndex& body_index,
                               double transverse_surface_area, double longitudinal_surface_area,
                               double transverse_CD, double longitudinal_CD,
                               const math::RigidTransform<double>& X_BodyCP =
                                   math::RigidTransform<double>::Identity(),
                               double fluid_density = kDefaultFluidDensity);

 private:
  // Calculates the spatial forces in the world frame as expected by the
  // applied_spatial_force input port of MultibodyPlant.
  void CalcSpatialForce(
      const systems::Context<T>& context,
      std::vector<ExternallyAppliedSpatialForce<T>>* spatial_force) const;

  // Calculates the aerodynamic center output port.
  void CalcAerodynamicCenter(const systems::Context<T>& context,
                             systems::BasicVector<T>* aerodynamic_center) const;

  // Declare friendship to enable scalar conversion.
  template <typename U>
  friend class SlenderBody;

  const BodyIndex body_index_;
  const math::RigidTransform<double> X_BodyCP_;
  const double transverse_surface_area_;
  const double longitudinal_surface_area_;
  const double transverse_CD_;
  const double longitudinal_CD_;
  const double default_fluid_density_;
};

}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class drake::multibody::SlenderBody)
