import React from 'react';
import Feature from '../../components/feature/Feature';
import './features.css';

const featuresData = [
  {
    title: '3D Spline Track Models',
    text: 'Full support of 3D modeling tracks ensures that you get the upper edge of accuracy in your calculations.',
  },
  {
    title: 'Optimized Racing Line Calculations',
    text: 'Low level optimized algorithms calculate optimal racing lines in every situation, pushing the limit of modern machinery.',
  },
  {
    title: 'Telemetry Data Analysis',
    text: 'Compare your own laps with our calculated laps to see beautifully analyzed telemtry.',
  },
  {
    title: 'AI Driving Corrections',
    text: 'Recieve AI generated driving suggestions at every part of the track. Gear changes, throttle rate, oversteer, etc.',
  },
];

const Features = () => (
  <div className="apex__features section__padding" id="features">
    <div className="apex__features-heading">
      <h1 className="gradient__text">Engineering has brought us near the edge of physical potential and adrenaline. Now AI can take us farther. </h1>
      <p>Performance Perfected.</p>
    </div>
    <div className="apex__features-container">
      {featuresData.map((item, index) => (
        <Feature title={item.title} text={item.text} key={item.title + index} />
      ))}
    </div>
  </div>
);

export default Features;