import math

import numpy as np
import taichi as ti
from pathlib import Path
vec3 = ti.math.vec3
dot = ti.Vector.dot

#region CONSTANTS
# feel free to change these (reasonable precision with citations might get you points!)
VIEW_RAY_STEPS = 16
SUN_RAY_STEPS = 4

EARTH_RADIUS_KM = 6000.0 # if you modify this make sure to update line 38 of demo.py

# Source: WGS84
EARTH_EQUATORIAL_RADIUS_KM = 6378.137
EARTH_POLAR_RADIUS_KM = 6356.752314245

EARTH_FLATTENING_FACTOR = (EARTH_EQUATORIAL_RADIUS_KM - EARTH_POLAR_RADIUS_KM) / EARTH_EQUATORIAL_RADIUS_KM

# Source: NASA Earth Fact Sheet
EARTH_SCALE_HEIGHT_KM = 8.5

ATMOSPHERE_THICKNESS_KM = 110 # The true atmosphere extends quite far, so we choose a point where it visually seems to end
ATMOSPHERE_EQUATORIAL_RADIUS_KM = EARTH_EQUATORIAL_RADIUS_KM + ATMOSPHERE_THICKNESS_KM
ATMOSPHERE_POLAR_RADIUS_KM = EARTH_POLAR_RADIUS_KM + ATMOSPHERE_THICKNESS_KM
# ^ this atmosphere is actually very innacurate, but sometimes it's instructive to have it be big to see what's going on
# be careful though, sometimes having a thick atmosphere leads to unintended consequences

# Source: Precomputed Atmospheric Scattering - Eric Bruneton, Fabrice Neyret
BETA_RAYLEIGH = vec3(3.8e-3, 13.5e-3, 33.1e-3)
#endregion

#region HELPER FUNCTIONS

# This function transforms the ray into a coordinate space where the z axis is scaled by width/height
@ti.func
def _convert_ray_to_sphere_space(origin: vec3, direction: vec3, width: ti.f32, height: ti.f32):
    w_over_h = width / height
    sphere_origin = vec3(origin.x, origin.y, origin.z * w_over_h)
    sphere_dir = vec3(direction.x, direction.y, direction.z * w_over_h)
    return sphere_origin, sphere_dir.normalized()

# This function transforms a point into a coordinate space where the z axis is scaled by height/width
@ti.func
def _convert_pos_to_spheroid_space(pos: vec3, width: ti.f32, height: ti.f32) -> vec3:
    return vec3(pos.x, pos.y, pos.z / (width / height))


# Given an origin and a ray, this function returns up to two points of intersection,
# along with booleans for whether the collision point exists
#
# I wonder why this function generalizes to spheroids?
@ti.func
def cast_ray_against_oblate_spheroid(origin: vec3, direction: vec3, width: ti.f32, height: ti.f32):
    sphere_origin, sphere_dir = _convert_ray_to_sphere_space(origin, direction, width, height)

    a = sphere_dir.dot(sphere_dir)
    b = 2.0 * sphere_origin.dot(sphere_dir)
    c = sphere_origin.dot(sphere_origin) - width * width
    determinant = (-4.0 * c * a) + (b * b)

    first_pos = vec3(9.0, 9.0, 9.0)
    collides_first = False
    second_pos = vec3(0.0, 0.0, 0.0)
    collides_second = False

    if determinant >= 0.0:
        sqrt_det = ti.sqrt(determinant)
        two_a = 2.0 * a
        small_t = (-b - sqrt_det) / two_a
        large_t = (-b + sqrt_det) / two_a

        first_pos = vec3(0.0, 0.0, 0.0)
        second_pos = vec3(0.0, 0.0, 0.0)

        if small_t >= 0.0:
            hit = sphere_origin + sphere_dir * small_t
            first_pos = _convert_pos_to_spheroid_space(hit, width, height)
            collides_first = True
        if large_t >= 0.0:
            hit = sphere_origin + sphere_dir * large_t
            second_pos = _convert_pos_to_spheroid_space(hit, width, height)
            collides_second = True

    return first_pos, collides_first, second_pos, collides_second
#endregion

# notice every function needs @ti.func so it can run in the kernel
@ti.func
def funnyFunction(pos, ray, sun_dir):
    atmosphereHitPosition, collided,  _, _ = cast_ray_against_oblate_spheroid(pos, ray, EARTH_RADIUS_KM+ATMOSPHERE_THICKNESS_KM, EARTH_RADIUS_KM+ATMOSPHERE_THICKNESS_KM)
    funGradient = vec3(0,0,0)
    if (collided):
        hit_dir = atmosphereHitPosition.normalized()
        # notice you can sometimes get weird behavior when there's negative values
        funGradient = (hit_dir*0.5+vec3(0.5,0.5,0.8)) * max(dot(hit_dir, sun_dir),0.05) + _earth(pos, ray, sun_dir)*vec3(0.5,0.5,0.7) + vec3(0.05,0.02,0.05)
    else:
        funGradient = (ray-vec3(0.5,0,0))*0.5+vec3(0.5,0.5,0.5)
    return funGradient


@ti.func
def _atmos(pos, ray, sun_dir):
    # Find rays intersections w/ atmosphere and Earth
    p_atmos_1, hit_a1, p_atmos_2, hit_a2 = cast_ray_against_oblate_spheroid(pos, ray, ATMOSPHERE_EQUATORIAL_RADIUS_KM, ATMOSPHERE_POLAR_RADIUS_KM)
    p_earth_1, hit_e1, p_earth_2, hit_e2 = cast_ray_against_oblate_spheroid(pos, ray, EARTH_EQUATORIAL_RADIUS_KM, EARTH_POLAR_RADIUS_KM)

    total_color = vec3(0.0)

    cos_theta = ray.dot(sun_dir)
    phase_rayleigh = 3.0 / (16.0 * ti.math.pi) * (1.0 + cos_theta * cos_theta)

    # Accumulates optical depth with each step along ray
    view_optical_depth = 0.0

    if hit_a1:
        start_pos = p_atmos_1
        end_pos = p_atmos_2

        # If ray hits Earth end at ground
        if hit_e1:
            end_pos = p_earth_1


        # Draw a vector from the start to the end
        segment = end_pos - start_pos
        ray_length = segment.norm()
        step_size = ray_length / VIEW_RAY_STEPS



        for i in range(VIEW_RAY_STEPS):
            # Use midpoint Riemann sum
            sample_pos = start_pos + ray * step_size * (float(i) + 0.5)
            sample_dir = sample_pos.normalized()

            # Calculate altitude using oblate spheroid Earth model
            local_earth_radius = EARTH_EQUATORIAL_RADIUS_KM * (1.0 - (sample_dir.z * sample_dir.z) * EARTH_FLATTENING_FACTOR)
            altitude = sample_pos.norm() - local_earth_radius
            altitude = ti.max(0.0, altitude)

            # Calculate density (Barometric Formula) and add to optical depth
            local_density = ti.exp(-altitude / EARTH_SCALE_HEIGHT_KM)
            view_optical_depth += local_density * step_size

            # Cast ray to find if sun is blocked by Earth
            _, sun_hit_e1, _, _ = cast_ray_against_oblate_spheroid(
                sample_pos, sun_dir, EARTH_EQUATORIAL_RADIUS_KM, EARTH_POLAR_RADIUS_KM
            )

            # If sun ray doesn't hit Earth, sample is not in shadow
            if not sun_hit_e1:
                # Find where sun ray exits the atmosphere
                _, _, p_sun_exit, hit_sun_exit = cast_ray_against_oblate_spheroid(
                    sample_pos, sun_dir, ATMOSPHERE_EQUATORIAL_RADIUS_KM, ATMOSPHERE_POLAR_RADIUS_KM
                )

                sun_ray_length = (p_sun_exit - sample_pos).norm()
                sun_step_size = sun_ray_length / SUN_RAY_STEPS
                sun_optical_depth = 0.0

                # March Sun Ray
                for j in range(SUN_RAY_STEPS):
                    sun_sample_pos = sample_pos + sun_dir * sun_step_size * (float(j) + 0.5)
                    sun_sample_dir = sun_sample_pos.normalized()

                    # Calculate altitude for the sun sample point
                    sun_local_radius = EARTH_EQUATORIAL_RADIUS_KM * (
                                1.0 - (sun_sample_dir.z * sun_sample_dir.z) * EARTH_FLATTENING_FACTOR)
                    sun_altitude = ti.max(0.0, sun_sample_pos.norm() - sun_local_radius)

                    # Add to sun optical depth
                    sun_local_density = ti.exp(-sun_altitude / EARTH_SCALE_HEIGHT_KM)
                    sun_optical_depth += sun_local_density * sun_step_size


                combined_view_depth = view_optical_depth + sun_optical_depth

                transmittance = ti.exp(-combined_view_depth * BETA_RAYLEIGH)

                # Calculate the final light added at this specific sample point
                scattered_light = local_density * step_size * transmittance * BETA_RAYLEIGH * phase_rayleigh

                total_color += scattered_light

    final_transmittance = ti.exp(-view_optical_depth * BETA_RAYLEIGH)
    return total_color, final_transmittance, vec3(0.0)

# Used by demo.py
# This earth function is very simple, it renders a blueish-grey ball
# Do not touch if you are competing for accuracy, unless you are changing earth's radius
# A simple earth helps the judges see your atmosphere more clearly
#
# Feel free to play with it if you're making an art piece
@ti.func
def _earth(pos, direction, sun_dir):
    surface_km, hit, _, _ = cast_ray_against_oblate_spheroid(pos, direction, EARTH_RADIUS_KM, EARTH_RADIUS_KM)
    color = vec3(0,0,0)
    if hit:
        surface_dir = surface_km.normalized()
        ndotl = max(surface_dir.dot(sun_dir),0)
        if ndotl >= 0.0:
            color = vec3(0.7,0.7,1)*ndotl*0.5
    return color
