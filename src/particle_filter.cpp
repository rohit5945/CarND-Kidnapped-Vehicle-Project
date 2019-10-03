/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::normal_distribution;
using std::string;
using std::vector;
void ParticleFilter::init(double x, double y, double theta, double std[])
{
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::default_random_engine gen;
  num_particles = 100; // TODO: Set the number of particles
  // Resize weights vector based on num_particles
  weights.resize(num_particles);

  // Resize vector of particles
  particles.resize(num_particles);
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  for (int i = 0; i < num_particles; ++i)
  {

    //double sample_x, sample_y, sample_theta;
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1.0;
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
  for (int i = 0; i < num_particles; ++i)
  {
    if (abs(yaw_rate) != 0)
    {
      particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
    else
    {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }

    // Add noise to the particles
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations)
{
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  double min_dist = std::numeric_limits<double>::max();
  std::vector<int> associations;
  std::vector<double> sense_x;
  std::vector<double> sense_y;
  int mapping_id = -1;
  for (int i = 0; i < observations.size(); ++i)
  {
    for (int j = 0; j < predicted.size(); ++j)
    {
      float distance = HELPER_FUNCTIONS_H_::dist(observations[i].x, observations[i].y, predicted[i].x, predicted[i].y);
      if (distance < min_dist)
      {
        min_dist = distance;
        mapping_id = predicted[i].id;
      }
    }
    observations[i].id = mapping_id;
    associations.push_back(mapping_id);
    sense_x.push_back(observations[i].x);
    sense_y.push_back(observations[i].y);
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  for (int i = 0; i < num_particles; ++i)
  {

    //vector to hold landmarks that are within sensor range
    std::vector<LandmarkObs> map_landmarks_within_sensor_range;

    //get all landmarks within sensor range
    for (int k = 0; k < map_landmarks.landmark_list.size(); k++)
    {
      float map_landmark_x = map_landmarks.landmark_list[k].x_f;
      float map_landmark_y = map_landmarks.landmark_list[k].y_f;

      if (fabs(map_landmark_x - particles[i].x) <= sensor_range && fabs(map_landmark_y - particles[i].y) <= sensor_range)
      {
        map_landmarks_within_sensor_range.push_back(LandmarkObs{map_landmarks.landmark_list[k].id_i, map_landmark_x, map_landmark_y});
      }
    }

    //convert all observations from VCS to map for particle[i]
    std::vector<LandmarkObs> transformed_obs;
    for (int j = 0; j < observations.size(); j++)
    {

      double transformed_x = particles[i].x + (cos(particles[i].theta) * observations[j].x) - (sin(particles[i].theta) * observations[j].y);

      // transform to map y coordinate
      double transformed_y = particles[i].y + (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y);

      //push this observation onto new list

      transformed_obs.push_back(LandmarkObs{observations[j].id, transformed_x, transformed_y});
    }

    dataAssociation(map_landmarks_within_sensor_range, transformed_obs);

    // reinit weight
    particles[i].weight = 1.0;
    for (int j = 0; j < transformed_obs.size(); j++)
    {
      double tr_obs_x, tr_obs_y, lm_in_senrange_x, lm_in_senrange_y;
      tr_obs_x = transformed_obs[j].x;
      tr_obs_y = transformed_obs[j].y;

      for (int k = 0; k < map_landmarks_within_sensor_range.size(); k++)
      {
        if (map_landmarks_within_sensor_range[k].id == transformed_obs[j].id)
        {
          lm_in_senrange_x = map_landmarks_within_sensor_range[k].x;
          lm_in_senrange_y = map_landmarks_within_sensor_range[k].y;
        }
      }
      // calculate weight for this observation with multivariate Gaussian
      double s_x = std_landmark[0];
      double s_y = std_landmark[1];
      double obs_w = 1 / (2 * M_PI * s_x * s_y) * exp(-0.5 * (pow((lm_in_senrange_x - tr_obs_x) / s_x, 2) + pow((lm_in_senrange_y - tr_obs_y) / s_y, 2)));
      if (obs_w > 0)
        particles[i].weight *= obs_w;
    }
    std::cout << "Particle weight :" << particles[i].weight << std::endl;
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample()
{

  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  std::vector<Particle> new_particles(num_particles);
  std::random_device rd;
  std::default_random_engine gen(rd());

  for (int i = 0; i < num_particles; i++)
  {
    std::discrete_distribution<int> index(weights.begin(), weights.end());
    new_particles[i] = particles[index(gen)];
  }
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}