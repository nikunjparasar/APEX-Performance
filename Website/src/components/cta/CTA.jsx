import React from 'react';
import physics from '../../assets/physics.png';

import './cta.css';

const CTA = () => (
  <div className="apex__cta">
    <div className="apex__cta-content">
      <div className="apex__cta-left">
        <img src={physics} alt="Track model" />
      </div>
      <div className="apex__cta-right">
        <h3>HOW IT WORKS.</h3>
        <p className='apex__cta-content_main'>GPS Modeled Track analysis for 25+ famous racetracks!</p>
        <p className='apex__cta-content_main'>Calculating the optimal racing line for a track involves a number of mathematical equations and physical calculations. To begin with, the track data in the form of CSV files need to be parsed and converted into a format that can be used for analysis. Next, cubic spline interpolation is used to smooth out the track data and generate a continuous function that can be used to calculate the ideal racing line. The physical calculations involved in this process include taking into account factors such as peak engine power, friction coefficients, inertia, drag, and more. Having an accurate tire model is essential for calculating the optimal racing line, as it allows the driver to push the car to its limits without losing control. A good tire model can help the driver to understand how the car will behave in different situations, such as under hard braking, during cornering, and when accelerating out of turns. These factors are used to determine the optimal speed and trajectory of the racing line, ensuring that the car can achieve the fastest possible lap time while also minimizing the risk of accidents and other issues. </p>
      </div>
    </div>
  </div>
);

export default CTA;
