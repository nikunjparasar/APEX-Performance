import React from 'react';
import DashGraph from './DashGraph';

import './cta.css';


const CTA = () => (
  <div className="apex__cta">
    <div className="apex__cta-content">
      <h3>Drag over a turn to zoom in and examine the racing line.</h3>
      <p>GPS Modeled Track analysis for 25+ famous racetracks!</p>
    </div>
    <div className="apex__cta-dash">
      <DashGraph />
    </div>
    

  </div>
);

export default CTA;