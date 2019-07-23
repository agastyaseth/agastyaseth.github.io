---
layout: post
published: false
title: Smart India Hackathon - Winner
---
## Tata Motors: Intelligent trip management for electric vehicles - SIH Hardware 2019

## Problem
Electric vehicles, especially pertinent to countries like India, suffer from a major disadvantage of not having charging stations as easily accessible as gas stations for regular cars. One of the biggest challenges in the adoption of electric cars lie in this range anxiety: one fears of being stranded on the road without having a charging station in the vicinity.

We were supposed to build a trip management system that would integrate in electric vehicles, keeping physical parameters in mind (dimensions of the vehicle, driver behavior etc.), along with various environmental disturbances like traffic conditions, headwinds etc., and would mitigate this range anxiety

## Our Solution
We decided to break-down the solution into three prongs:

* **Range Prediction**: Given the source and destination, it should be able to tell whether or not the vehicle would be able to make the trip in the charge available.
* **Internal Optimization**: Tuning the internal parameters of the vehicle and reduce auxiliary power consumption to increase this predicted range of the vehicle
* **Route planning**:  For trips that cannot be completed even after internal optimizations, or for longer trips , the user must be provided with a route that routes him/her through charging stations to the destination.

To solve the first point, we decided to use a Simulink® model of an electric vehicle, which would factor in the various force coefficients and dimensions of the car. When the driver enters the source and destination, using real-time traffic API provided by Tom Tom®, we would populate a section-wise velocity/time graph of the trip. This velocity/time graph when provided as input to the Simulink® model would give a range approximation, answering our question on whether or not we’d be able to make the trip.

For point two, we found literature to corroborate that most of the auxiliary power consumed by a vehicle can be attributed to the AC and the infotainment systems. We devised an algorithm that would optimize these systems keeping in mind the ANSI/ASHRAE Standard 55 for cooling and human comfort. 

More pertinent to buses and public transport, we also implemented an occupancy detector using deep learning models to dynamically tune the parameters like AC and lighting, depending on the occupancy of the bus.

Lastly, in case the vehicle still fails to make the trip, we used Google Maps API to recursively route the driver through optimal charging points to help him plan longer trips.
