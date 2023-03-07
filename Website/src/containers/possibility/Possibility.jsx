import React from 'react';
import videopromo from '../../assets/video.mp4';

import './possibility.css';

const Possibility = () => (
  <div className="apex__possibility section__padding" id="possibility">
    
    <div className="apex__possibility-image">
      <video autoPlay = {true} loop muted className='video' >
        <source src={videopromo} type="video/mp4"/>
      </video>
    </div>
    <div className="apex__possibility-content">
      <h4>Request Early Access to Get Started</h4>
      <h1 className="gradient__text">The possibilities are <br /> beyond your imagination</h1>
      <p>Experience cutting edge performance and dynamics through the vivid detail in which APEX visualizes and processes your data. Reach new limits and break new records.</p>
    </div>
  </div>
);

export default Possibility;