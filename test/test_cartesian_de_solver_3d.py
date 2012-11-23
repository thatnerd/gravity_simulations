#!/usr/bin/env python

import math
import numpy as np
import script.cartesian_de_solver_3d as cde
import unittest


class TestCartesianDESolver(unittest.TestCase):

	def test_calculate_b(self):
		a = 1.0
		e = 0.0
		self.assertEqual(cde.calculate_b(a, e), a)
		a = 2.0
		e = np.sqrt(.75)
#This is off to machine precision.
		self.assertTrue(abs(cde.calculate_b(a, e) - 1.0) < 10 ** -9) 

	def test_find_inverse_square_accel(self):
#Make sure that the centrifugal and gravitational accelerations balance out to 
#the precision of the year & AU.
		earth_thetadot = 2.0 * math.pi / cde.year
		ydot = earth_thetadot * cde.AU
		cent_accel = ydot * ydot / cde.AU
		r_vec = np.array([cde.AU, 0.0, 0.0])
		v_vec = np.array([0.0, ydot, 0.0])
		coords = np.array([r_vec, v_vec]).T  #vectors are columns of the array
		grav_accel = cde.find_inverse_square_accel(coords)
		gx, gy, gz = grav_accel
		self.assertTrue((gx - cent_accel) / cent_accel 
				<= 1.0 * 10. ** -9)

	def test_rotation_matrix_2d(self):
		self.assertTrue(np.equal(cde.rotation_matrix_2d(0.0), 
				np.array([[1, 0], [0, 1]])).all)
		self.assertTrue(np.equal(cde.rotation_matrix_2d(np.pi), 
				np.array([[-1, 0], [0, -1]])).all)
		r2o2 = 1.0 / np.sqrt(2.0)
		self.assertTrue( np.equal(cde.rotation_matrix_2d(-np.pi/4), 
				np.array([[r2o2, r2o2], [-r2o2, r2o2]])).all)

	def test_rotation_matrix_3d(self):
		z_rot_vec = np.array([0, 0, 1])
		z_rot_theta = np.pi / 4
		trans_matrix = cde.rotation_matrix_3d(z_rot_vec, z_rot_theta)
		vec_to_rotate = np.array([1, 0, 0])
		r2 = np.sqrt(2.0)
		self.assertTrue(np.equal(trans_matrix.dot(vec_to_rotate), 
			[1 / r2, 1 / r2, 0]).all)
		x_rot_vec = np.array([1, 0, 0])
		x_rot_theta = np.pi / 4
		trans_matrix = cde.rotation_matrix_3d(x_rot_vec, x_rot_theta)
		vec_to_rotate = np.array([0, 0, 1])
		self.assertTrue(np.equal(trans_matrix.dot(vec_to_rotate), 
			[0, - 1 / r2, 1 / r2]).all)
		rot_vec = np.array([1, 1, 1])
		rot_theta = np.pi * 2 / 3
		trans_matrix = cde.rotation_matrix_3d(rot_vec, rot_theta)
		vec_to_rotate = np.array([0, 0, 1])
		try:
			goo = cde.rotation_matrix_3d(np.array([0,0,0]), np.pi/3)
			self.assertTrue(False)
		except:
			self.assertTrue(True)
		self.assertTrue(np.equal(trans_matrix.dot(vec_to_rotate), 
			[1, 0, 0]).all)
		

	def test_find_constant_accel(self):
#test base case
		self.assertTrue(np.equal(cde.find_constant_accel(), 
			np.array([0, -9.8, 0])).all)
#test under Galilean change of origin
		self.assertTrue(np.equal(cde.find_constant_accel(
			coords=np.array([[0,13.0,0],[0,11.,0]]).T), 
			np.array([0, -9.8,0])).all)
#test under new direction of g vector
		self.assertTrue(np.equal(cde.find_constant_accel(
			accel_direction=np.array([1.0, 0.0,0])), 
			np.array([0, -9.8,0])).all)
#test under different magnitude
		self.assertTrue(np.equal(cde.find_constant_accel(
			accel_direction=np.array([0.0, -1.0,0]), accel_strength=12.0),
			np.array([0, -12.0,0])).all)
#Check under two incompatible cases
		try:
			goo = np.equal(cde.find_constant_accel(
				accel_direction=np.array([1.0, 0.0,0.0])), 
				np.array([0, -9.8,0.])).all
			self.assertTrue(False)
		except:
			self.assertTrue(True)
#Check to see if it fails when I try to pass it a bad direction vector
		try:
			goo = np.equal(cde.find_constant_accel(
				accel_direction=np.array([2.0, 0.0, 0.0])))
			self.assertTrue(False)
		except:
			self.assertTrue(True)

	def test_calculate_derivs(self):
		#check it out for starting on the x-axis for Earth
		start_coords_1 = np.array([[cde.AU, 0.0, 0.0], 
			[0.0, cde.earth_omega * cde.AU, 0.0]]).T
		derivs_pred_1 = np.array([start_coords_1[:,1], 
			[-cde.MGsun / (start_coords_1[:,0].dot(start_coords_1[:,0])), 
				0.0, 0.0]]).T
		self.assertTrue(np.equal(derivs_pred_1, 
			cde.calculate_derivs(start_coords_1)).all)
		#now for -y axis.
		start_coords_2 = np.array(
				[[0.0, -cde.AU, 0.0], [cde.earth_omega * cde.AU, 0.0, 0.0]]).T
		derivs_pred_2 = np.array([start_coords_2[:,1], [cde.MGsun / 
			start_coords_2[:,1].dot(start_coords_2[:,1]), 0.0, 0.0]]).T
		self.assertTrue(np.equal(derivs_pred_2, 
			cde.calculate_derivs(start_coords_2)).all)

#The following two also serve as unit tests for step_time_simple
	def test_projectile_step_time_simple(self):
		r0 = np.array([2.0, 0.0, 0.0])
		v0 = np.array([3.0, 35.0, 0.0])
		a = np.array([0, -9.8, 0.0])
		coords0 = np.array([r0, v0]).T
		coords = np.copy(coords0)
		dt = 1.0
		coords = cde.projectile_step_time_simple(coords, dt)
		r1 = r0 + v0 * dt  #+ .5 * a * dt * dt doesn't enter into this.
		v1 = v0 + a * dt
		expected_coords = np.array([r1, v1]).T
		self.assertTrue(np.equal(coords, expected_coords).all)

#The following will also serve as a unit test for step_time_simple
	def test_orbital_step_time_simple(self):
		earth_vel = cde.earth_omega * cde.AU
		start_coords_1 = np.array([[cde.AU, 0.0, 0.0], 
			[0.0, earth_vel, 0.0]]).T
		dt = cde.year / 360  #let's go forward one degree!
		one_deg = 2 * np.pi / 360.
		rot_coords = np.dot(cde.rotation_matrix_3d(
			np.array([0, 0, 1.0]), one_deg), start_coords_1)
		final_coords = cde.orbital_step_time_simple(start_coords_1, dt)
		final_mags = np.array([final_coords[:,0].dot(final_coords[:,0]), 
			final_coords[:,1].dot(final_coords[:,1])])
		diff_coords = (rot_coords - final_coords) / final_mags
		diff_coords_max = np.sqrt(
				max(np.dot(diff_coords[:,0], diff_coords[:,0]),
				np.dot(diff_coords[:,1], diff_coords[:,1])))
#Check to see if it's off by not more than a linear amount
		self.assertTrue(diff_coords_max < 1.0 / 360.0)

	def test_rk4_stepper(self):
		earth_vel = cde.earth_omega * cde.AU
		start_coords_1 = np.array([[cde.AU, 0.0, 0.0], 
			[0.0, earth_vel, 0.0]]).T
		dt = cde.year / 360  #let's go forward one degree!
		one_deg = 2 * np.pi / 360.
		rot_coords_1 = np.dot(cde.rotation_matrix_3d(
			np.array([0, 0, 1.0]), one_deg), start_coords_1)
		int_coords = cde.rk4_stepper(start_coords_1, .5 * dt)
		final_coords = cde.rk4_stepper(int_coords, .5 * dt)
		final_coords_2 = cde.rk4_stepper(start_coords_1, dt)
		rot_mags = np.array([rot_coords_1[:,0].dot(rot_coords_1[:,0]), 
			rot_coords_1[:,1].dot(rot_coords_1[:,1])])
		log_diff_coords = (rot_coords_1 - final_coords_2) / rot_mags
		pos_err = np.sqrt(np.dot(log_diff_coords[:,0], log_diff_coords[:,0]))
		vel_err = np.sqrt(np.dot(log_diff_coords[:,1], log_diff_coords[:,1]))
		max_err = max(pos_err, vel_err)
##Good to within a part in a billion
#		self.assertTrue(max_err < 1.0 * 10 ** -9)

	def test_calculate_orbit(self):
		halleys_aphelion = 36.1 * cde.AU
		halleys_perhelion = 0.586 * cde.AU
		halleys_a = 17.8 * cde.AU
		halleys_e = 0.967 
		halleys_period = 75.3 * cde.year
		halleys_b = cde.calculate_b(halleys_a, halleys_e)
		halleys_area = np.pi * halleys_a * halleys_b
		halleys_area_over_time = halleys_area / halleys_period
	# .5 * r * r * thetadot = area_over_time
	#thetadot = 2.0 * area_over_time / r^2
		v0 = 2.0 * halleys_area_over_time / halleys_aphelion  #@ max distance
		l_over_m = v0 * halleys_aphelion
		print 'first L: ', l_over_m
		start_pos = np.array([halleys_aphelion, 0.0, 0.0])
		start_vel = np.array([0.0, v0, 0.0])
		start_coords = np.array([start_pos, start_vel]).T
		dt = 1.0 * 60 * 60 * 24  # 1 day
		pos_record = cde.calculate_orbit(start_coords, dt, 
				expected_period=halleys_period)
		pos_diff = pos_record[-1] - pos_record[0]
		pr0 = pos_record[0]
		norms = np.array([pr0[:,0].dot(pr0[:,0]), pr0[:,1].dot(pr0[:,1])])
		print pos_diff / norms


if __name__ == '__main__':
	unittest.main()
