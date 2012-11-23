#!/usr/bin/env python
import numpy as np
import script.cartesian_de_solver_3d as cde





def main():
	comet_data = 'doc/halleys_comet_orbit.csv'  #Store the data here!
	halleys_aphelion = 35.1 * cde.AU
	halleys_perhelion = 0.586 * cde.AU
	halleys_a = 17.8 * cde.AU
	halleys_e = 0.967 
	halleys_period = 75.3 * cde.year
	halleys_b = cde.calculate_b(halleys_a, halleys_e)
	halleys_area = np.pi * halleys_a * halleys_b
	halleys_area_over_time = halleys_area / halleys_period
## .5 * r * r * thetadot = area_over_time
##thetadot = 2.0 * area_over_time / r^2
	v0 = 2.0 * halleys_area_over_time / halleys_aphelion  #@ max distance
	l_over_m = v0 * halleys_aphelion
	print 'first L: ', l_over_m
	start_pos = np.array([halleys_aphelion, 0.0, 0.0])
	start_vel = np.array([0.0, v0, 0.0])
	start_coords = np.array([start_pos, start_vel]).T
	dt = 1.0 * 60 * 60 * 24  # 1 day
	coord_record = cde.calculate_orbit(start_coords, dt, 
			expected_period=halleys_period + 0.1 * cde.year)
	print 'num data points =', len(coord_record)

	import csv
	my_file = open(comet_data, 'w')
	my_writer = csv.writer(my_file)
	my_writer.writerow(['x', 'y', 'z', 'v_x', 'v_y', 'v_z'])
	for row in coord_record:
		my_writer.writerow(row.T.flat[:])
	my_file.close()  #doesn't read all lines if you don't do this.
	import matplotlib.mlab as mlab
	my_data = mlab.csv2rec(comet_data)
	print 'number of points after reading csv:', len(my_data)
	print "Halley's actual semimajor axis: (good to 3 digits)", halleys_a
	my_a = (np.max(my_data['x']) + np.min(my_data['x'])) / 2.0
	print "My calculated semimajor axis: ", my_a
	print "Logarithmic discrepancy:", (halleys_a - my_a) / halleys_a
	print "Expected logarithmic discrepancy: ", 0.1 * cde.AU / halleys_a
	print 'my data = ', my_data[0]
	print type(my_data[0])
	print 'dt = ', dt
	period, final_coords = cde.find_orbit_completion_point(my_data, dt)
	print 'period = ', period
	print 'final_coords = ', final_coords
	print 'diff = ', final_coords - start_coords
	print 'x ratio =', (final_coords[0][0] - start_coords[0][0]) /\
			final_coords[0][0]
	print 'v_y ratio =', (final_coords[1][1] - start_coords[1][1]) /\
			final_coords[1][1]
#	import matplotlib.pyplot as plt
#	my_plot = plt.figure()
#	ax = my_plot.add_subplot(111)
#	plt.scatter(my_data['x'], my_data['y'], s=1, facecolor='0.5', lw=0)
#	plt.show()


if __name__ == '__main__':
	main()
