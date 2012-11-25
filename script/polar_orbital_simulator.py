#!/usr/bin/env python
"""
This is a bunch of code which definitely does not work. Don't use it. It can't
even model a straight line.
"""


import math

AU =   1.49598261 * 10.0 ** 11  #meters, 9 digits
year = 3.65256363 * 100.0 * 24.0 * 60.0 * 60.0  #seconds, 9 digits
MGsun = 4 * math.pi * math.pi * AU * AU * AU / (year * year)  #9 digits


def find_rddot(r, thetadot, MG=MGsun):
	return (- MG / (r * r)) + r * thetadot * thetadot


def step_time_simple(r0, theta0, rdot0, L_over_m, dt, MG=MGsun):
	"""
	Relatively simple differential equation stepper. 
	Angular momentum (L = r * thetadot) is conserved
	Second derivatives are assumed to be constant
	
	Inputs
	------
	r0, theta0, rdot0, L_over_m, dt, M: all floats. Dots represent partial 
		derivatives wrt time.
	
	Outputs
	-------
	r1, theta1, rdot1: all floats of the values that the variables take 
		after the time dt. 
	"""
	thetadot0 = L_over_m / r0 / r0
	ke_per_kg = 0.5 * L_over_m * L_over_m / (r0 * r0)
	u_per_kg = - MG / r0
	e0 = u_per_kg + ke_per_kg
	rddot = find_rddot(r0 + .5 * rdot0 * dt, thetadot0, MG=MG)  #roughly avg r
#thetaddot is irrelevant; thetadot determined by conservation of momentum
	rdot1 = rdot0 + rddot * dt
	r1 = r0 + .5 * (rdot0 + rdot1) * dt
	thetadot1 = L_over_m / (r1 * r1)
	theta1 = theta0 + 0.5 * (thetadot0 + thetadot1) * dt
	return r1, theta1, rdot1


def main():
	pass


if __name__ == '__main__':
	main()
