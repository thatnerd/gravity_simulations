#!/usr/bin/env python
import numpy as np

AU =   1.49598261 * 10.0 ** 11  #meters, 9 digits
year = 3.65256363 * 100.0 * 24.0 * 60.0 * 60.0  #seconds, 9 digits
#standard gravitational parameter of the sun
MGsun = 4 * np.pi * np.pi * AU * AU * AU / (year * year)  #9 digits
earth_omega = 2.0 * np.pi / year


def calculate_b(a, e):
	"""
	I didn't test this, I just plugged in the formula, but it gives the right 
	answer.
	Given the semimajor axis and eccentricity of an ellipse, finds the 
	semiminor axis.

	Inputs
	------
	a: the length of the semimajor axis
	e: the eccentricity of the ellipse

	Returns
	-------
	b: the length of the semiminor axis
	"""
	return np.sqrt(a * a * (1 - e * e))


def find_inverse_square_accel(coords, strength=MGsun, 
		grav_origin=np.array([0.0,0.0,0.0])):
	"""
	Find the acceleration due to gravity

	Inputs
	------
	coords: numpy array([two 2D vectors]), 2x2. The first vector (coords[:,0]) 
		representing the 2D position of the satellite or planet in the plane of
		the orbit, the second (coords[:,1]) representing its velocity.
	strength: Gravitational strength of the object (usually the sun)
	grav_mass_location: also a numpy array. If the massive body generating the 
		field (usually the sun) is not at the origin, put this in.

	Returns
	-------
	g_vec: array([g_x, g_y]). Components of the gravitational acceleration 
		along the x and y coordinates
	"""
	rvec = coords[:,0]
	r_diff_vec = rvec - grav_origin
	r_squared = np.dot(r_diff_vec, r_diff_vec)
	r_mag = np.sqrt(r_squared)
	g_mag = strength / r_squared
	g_vec = (- g_mag / r_mag) * r_diff_vec
	return g_vec


def rotation_matrix_2d(theta):
	"""
	Generates a numpy matrix that, when multiplied by a right vector, rotates 
	it by the angle, theta, where theta is positive. Use 
	np.dot(output, np.array_of_shape(2)) to actually rotate it. This is the 
	equivalent of rotating the coordinates by -theta while leaving the vector 
	itself the same.

	Inputs
	------
	theta: An angle, given in radians

	Returns
	-------
	np.array object of shape = (2, 2) that will multiply a column vector on the
	right.
	"""
	cs = np.cos(theta)
	sn = np.sin(theta)
	return np.array([[cs, -sn], [sn, cs]])


def rotation_matrix_3d(vector, theta):
	"""
	Generates a numpy matrix that, when multiplied by a right vector, rotates 
	it by the angle, theta, about the axis. Use 
	np.dot(output, np.array_of_shape_3) to actually rotate it. This is the 
	equivalent of rotating the coordinates by -theta while leaving the vector 
	itself the same.

	Inputs
	------
	vector: an np.array([x, y, z]) vector whose length is irrelevant 
		(though it can't be 0)
	theta: An angle, given in radians

	Returns
	-------
	np.array object of shape = (3, 3) that will multiply a column vector to the
		right with return.dot(vector_column), rotating it about vector (in a 
		right-handed mannder).
	"""
	vec_squared = vector.dot(vector)
	if vec_squared == 0:
		raise Exception("Cannot rotate about a vector of " + \
				"zero magnitude.")
	cs = np.cos(theta)
	sn = np.sin(theta)
	x, y, z = vector
	xsign = np.sign(x)
	xy_mag = vector[:2].dot(vector[:2])
	if xy_mag != 0:
		z_rot_theta = np.arcsin(y / xy_mag)
	else:
		z_rot_theta = 0.0
	if xsign < 0:
		z_rot_theta = np.pi - z_rot_theta  
	csz = np.cos(z_rot_theta)
	snz = np.sin(z_rot_theta)

#First, rotate about the z axis
	first_rotation = np.array([[csz, snz,0.0],[-snz, csz, 0.0], [0.0,0.0,1.0]])
	y_rot_theta = np.arccos(z / vector.dot(vector))
	csy = np.cos(y_rot_theta)
	sny = np.sin(y_rot_theta)
#Second, rotate around the rotated y axis
	second_rotation = np.array([[csy, 0.0, sny], [0.0, 1.0, 0.0], 
		[-sny, 0.0, csy]])
	third_rotation = np.array([[cs, -sn, 0.0], [sn, cs, 0.0], [0.0, 0.0, 1.0]])
	all_operations = [first_rotation, second_rotation, third_rotation, 
			np.linalg.inv(second_rotation), np.linalg.inv(first_rotation)]
	total_transform = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 
			[0.0, 0.0, 1.0]])
	for each_rotation in all_operations:
		total_transform = each_rotation.dot(total_transform)
	return total_transform


def find_constant_accel(coords=None, accel_direction=np.array([0.0,-1.0, 0.0]),
		accel_strength=9.80):
	"""
	Inputs
	------
	coords: put here for consistency with other vector fields, the force is 
		independent of both position and velocity.
	accel_direction: if you want the force to point in another direction, put a
		vector here. 
	accel_strength: (float) The magnitude of the acceleration vector.

	Returns
	-------
	a vector representing the x and y components of the acceleration
	"""
	if accel_strength < 0:
		raise Exception("Your accel_strength should be greater " +
				"than zero")
	direc_mag_2 = np.dot(accel_direction, accel_direction)
	if direc_mag_2 != 1.0:
		accel_direction =  accel_direction / np.sqrt(direc_mag_2)
	return accel_strength * accel_direction


def calculate_derivs(coords, accel=find_inverse_square_accel, *args, **kwargs):
	"""
	For a set of coordinates, this calculates their derivatives. The second 
		derivatives use calculate_g_vector (or accel).

	Inputs
	------
	coords: np.array([x, y, dx/dt, dy/dt])

	Returns
	-------
	derivs: np.array([dx/dt, dy/dt, d2x/dt2, d2y/dt2])
	"""
	accels = accel(coords, *args, **kwargs)
	derivs = np.array([coords[:,1], accels[:]]).T
# [[xdot, xddot], [ydot, yddot], [zdot, zddot]]
	return derivs


def step_time_simple(coords, dt, accel=find_inverse_square_accel, *args, 
		**kwargs):
	"""
	Simplest method of stepping forward in time for a differential equation, 
	this just changes position by velocity * dt, and then changes velocity by 
	acceleration * dt

	Inputs
	------
	coords: np.array([[x, y], [v_x, v_y]]).T
	dt: double representing the time step (in seconds)
	accel: The call to a function that returns the acceleration for a given 
		set of coords
	
	Returns
	-------
	an object of the same type as coords; essentially, it updates coords with
	'coords = step_time_simple(coords, dt, **kwargs)'
	"""
	return coords + dt * calculate_derivs(coords, accel=accel, *args, **kwargs)


def projectile_step_time_simple(coords, dt, 
		accel_direction=np.array([0.0,-1.0, 0.0]), accel_strength=9.80):
	"""
	Just an instance of step_time_simple that uses constant acceleration, with 
	appropriate defaults for g = (0.0, -9.8 m/s^2).
	"""
	return step_time_simple(coords, dt, accel=find_constant_accel, 
			accel_direction=accel_direction, accel_strength=accel_strength)


def orbital_step_time_simple(coords, dt, strength=MGsun, 
		grav_origin=(0.0, 0.0, 0.0)):
	"""
	Just an instance of step_time_simple that uses 1/r^2 acceleration, and 
	assumes we're dealing with something that's orbiting the sun, with nothing
	else in the solar system to worry about, and some simple defaults that 
	show this.
	"""
	return step_time_simple(coords, dt, accel=find_inverse_square_accel, 
			strength=strength, grav_origin=grav_origin)
#	return coords + dt * calculate_derivs(coords, strength=strength, 
#			grav_origin=grav_origin)


def rk4_stepper(coords, dt, accel=find_inverse_square_accel, *args, **kwargs):
	"""
	For an object at phase space coords, steps its position forward a time dt 
		in the presence of a force field (defaults to invers square gravity). 
	Uses Runge-Kutta 4 technique.
	If calculating an orbit, keep in mind that neither angular momentum nor 
		energy is conserved, so over time, errors will tend to add up. 
	Only works in 2D atm.
	
	Inputs
	------
	coords: np.array([[x, y], [dx/dt, dy/dt]]).T
	dt: amount of time you want to step forward by
	MG: strength of the gravitational field
	grav_origin: np.array([x, y]) of the origin of the gravitational field.

	Returns
	-------
	np.array([x, y], [dx/dt, dy/dt]).T (rather like coords)
	"""
	k1 = dt * calculate_derivs(coords, accel=accel, *args, **kwargs)
	k2 = dt * calculate_derivs(coords + (.5 * k1), accel=accel, *args, 
			**kwargs)
	k3 = dt * calculate_derivs(coords + (.5 * k2), accel=accel, *args, 
			**kwargs)
	k4 = dt * calculate_derivs(coords + k3, accel=accel, *args, **kwargs)
	return coords + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def calculate_orbit(initial_coords, dt, strength=MGsun, 
		grav_origin=np.array([0.0, 0.0, 0.0]), expected_period=None):
	"""
	Calculates an orbit for an object placed in the vicinity of the sun. Stops
	when it reaches a time equal to expected_period (or tries to estimate an 
	appropriate value)

	Inputs
	------
	initial_coords: np.array([[x, y], [v_x, v_y]]).T
	dt: float that gives the step size, in seconds
	expected_period: How long you want the simulator to run for; if None, 
		tries to estimate it from L and KE dimensionally

	Returns
	-------
	A list of the various values of initial_coords[:,0] at each step (ie, the 
		x and y positions)
	"""
	coords = np.copy(initial_coords)
	if expected_period is None:
		initial_pos = initial_coords[:,0]
		initial_vel = initial_coords[:,1]
		l_over_m = np.cross(initial_pos, initial_vel)
		print 'second L: ', l_over_m
		initial_vel_squared = initial_vel.dot(initial_vel)
		initial_r = np.sqrt(initial_pos.dot(initial_pos))
		initial_ke = .5 * initial_vel_squared #- strength / initial_r
		expected_period = .2 * l_over_m / initial_ke  #Just a guess.
#Honestly, I have no good idea of how to predict the period.
	
	time = 0.0
	coord_record = []
	year_count = 0.0
	coords_diff = 1.1 * coords
	while time <= expected_period:
		coord_record.append(coords)
		time += dt
		coords = rk4_stepper(coords, dt, accel=find_inverse_square_accel,
				strength=strength, grav_origin=grav_origin)

#Uncomment if you want a running tally of the number of years that have gone
#by.
#		if time > (year_count + 1) *  year:
#			print year_count, 'years', '(' + str(expected_period / year) + \
#					' expected)'
#			year_count += 1 * dt / year
	return coord_record


def find_orbit_completion_point(recarray_of_coords, dt, starting_row=27500):
	"""
	Finds the point in time where the position is closest to its original 
		position. Only works if we start with y0 = 0 and v_x0 = 0. Yes, it's a 
		oneoff.
	
	Inputs
	------
	recarray_of_coords: Should have keys 'x', 'y', 'v_x', and 'v_y'.
	Returns the time it took, plus the change in the various coords.

	Returns
	-------
	period: a number representing the number of seconds in the period of 
		rotation.
	final_coords: np.array([[x, v_x], [y, v_y]])
	"""
	list_of_coords = []
	for row in recarray_of_coords:
		new_coord = np.array([[row['x'], row['v_x']],[row['y'], row['v_y']]])
		list_of_coords.append(new_coord)

	start_coords = list_of_coords[0]
	x0, y0, vx0, vy0 = start_coords.T.flat[:]

	l_over_m = np.cross(list_of_coords[0][:,0], list_of_coords[0][:,1])
	if l_over_m >= 0: 
		parity = 1
	else:
		parity = -1
	prev_coords = list_of_coords[starting_row]
	step_count = starting_row
	delta_t = None
	final_coords = None
	loop_count = 0
	for coords in list_of_coords[starting_row + 1:]:
		loop_count += 1
		if loop_count % 10000 == 0:
			print "looped", loop_count, "times."
		step_count += 1
		xp, yp, vxp, vyp = prev_coords.T.flat[:]
		x, y, vx, vy = coords.T.flat[:]
		if parity * yp < y0 < parity * y:
			delta_t = parity * (y - y0) / (.5 * (vy + vyp))
			final_coords = rk4_stepper(prev_coords, delta_t)
			break
		prev_coords = coords
	if delta_t is not None and final_coords is not None:
		period = dt * step_count + delta_t
		return period, final_coords
	else:
		return None


def main():
	comet_data = 'doc/halleys_comet_orbit.csv'  #Store the data here!
	halleys_aphelion = 35.1 * AU
	halleys_perhelion = 0.586 * AU
	halleys_a = 17.8 * AU
	halleys_e = 0.967 
	halleys_period = 75.3 * year
	halleys_b = calculate_b(halleys_a, halleys_e)
	halleys_area = np.pi * halleys_a * halleys_b
	halleys_area_over_time = halleys_area / halleys_period
## .5 * r * r * thetadot = area_over_time
##thetadot = 2.0 * area_over_time / r^2
	v0 = 2.0 * halleys_area_over_time / halleys_aphelion  #@ max distance
	l_over_m = v0 * halleys_aphelion
	print 'first L: ', l_over_m
	start_pos = np.array([halleys_aphelion, 0.0])
	start_vel = np.array([0.0, v0])
	start_coords = np.array([start_pos, start_vel]).T
	dt = 1.0 * 60 * 60 * 24  # 1 day
	coord_record = calculate_orbit(start_coords, dt, 
			expected_period=halleys_period + 0.1 * year)

	import csv
	my_writer = csv.writer(open(comet_data, 'w'))
	my_writer.writerow(['x', 'y', 'v_x', 'v_y'])
	for row in coord_record:
		my_writer.writerow(row.T.flat[:])
	import matplotlib.mlab as mlab
	my_data = mlab.csv2rec(comet_data)
	print "Halley's actual semimajor axis: (good to 3 digits)", halleys_a
	my_a = (np.max(my_data['x']) + np.min(my_data['x'])) / 2.0
	print "My calculated semimajor axis: ", my_a
	print "Logarithmic discrepancy:", (halleys_a - my_a) / halleys_a
	print "Expected logarithmic discrepancy: ", 0.1 * AU / halleys_a
	period, final_coords = find_orbit_completion_point(my_data, dt)
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
