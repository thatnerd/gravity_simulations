#!/usr/bin/env python

import math
import script.polar_orbital_simulator as pos
import unittest

class TestPolarOrbitalSimulator(unittest.TestCase):

	def test_find_rddot(self):
#Make sure that the centrifugal and gravitational accelerations balance out to 
#the precision of the year & AU.
		earth_thetadot = 2.0 * math.pi / pos.year
		rddot = abs(pos.find_rddot(pos.AU, earth_thetadot)) 
		r_cent_accel = earth_thetadot * earth_thetadot * pos.AU
		self.assertTrue(rddot / r_cent_accel <= 1.0 * 10. ** -9)

	def test_step_time_simple(self):
		earth_thetadot = 2.0 * math.pi / pos.year
		earth_l_over_m = pos.AU * pos.AU * earth_thetadot
		step = 1.0 * 60  # 1 min
		theta = 0.0
		time_passed = 0.0
		r = pos.AU
		rdot = 0.0
		while time_passed < pos.year:
			r, theta, rdot = pos.step_time_simple(
					r, theta, rdot, earth_l_over_m, step)
			time_passed += step
		err = abs((theta - 2.0 * math.pi)) / (2.0 * math.pi)
		print err
		self.assertTrue(err <  1.0 / 365 / 24 / 60)
#actually goes around once per year for a circular orbit.

#		halleys_aphelion = 35.1 * pos.AU  
#		halleys_semimajor = 17.8 * pos.AU
#		halleys_period = 75.3 * pos.year
#		halleys_l_over_m = 2.0 * math.pi * \
#				halleys_semimajor * halleys_semimajor / halleys_period
#		min_approach = halleys_semimajor
#		r = halleys_aphelion
#		rdot = 0.0
#		theta = 0.0
#		time_passed = 0.0
#		count = 0
#		minute = 60.
#		hour = 60. * minute
#		day = 24. * hour
#		month = 30. * day
#		year = 365. * day
#		step = pos.year / 365  #3 days
#		while theta < 2.0 * math.pi:
#			if count % (20 * 365) == 0:
#				print count / 365 , 'years'
#				print '  theta = ', theta
#				print '  r = ', r / pos.AU, 'AU'
#				print '  rdot per year = ', rdot * year / pos.AU
#				thetadot = halleys_l_over_m / (r * r)
#				print '  thetadot per year = ', thetadot  * year
#				print '  a_r = ', pos.find_rddot(r, thetadot) * year * year / \
#						(pos.AU * pos.AU)
#				print '  centrif = ', halleys_l_over_m * halleys_l_over_m / \
#						(r * r * r) * year * year
#				print '  triangle area = ', thetadot * r * r
#			r, theta, rdot = pos.step_time_simple(r, theta, rdot, 
#					halleys_l_over_m, dt=step)
#			time_passed += step
#			if r < min_approach:
#				min_approach = r
#			count += 1
#		print 'period = ', time_passed / 60 / 60 / 24 / 365, 'years' 
#		print 'min_approach = ', min_approach / pos.AU, 'AU'

#Now, for a straight line:
		zero_MG = 0.0
		r = 10.0 ** 6  #m
		v0 = 1.0  #m/s 
		l_over_m = r * v0
		theta = 0.0
		rdot = 0.0
		step = 0.1  #seconds
		ttotal = 0.0
		count = 0
		while ttotal <= 100:
			r, theta, rdot = pos.step_time_simple(r, theta, rdot, l_over_m, 
					step)
			if count % 100 == 0:
				x = r * math.cos(theta)
				print 'x = ', x, 't = ', ttotal
			ttotal += step
			count += 1




if __name__ == '__main__':
	unittest.main()
