import React from 'react';
import DashGraph from './DashGraph';

import './cta.css';


const CTA = () => (
  <div className="apex__cta">
    <div className="apex__cta-content">
      <p>Drag over a turn to get started</p>
      <h3>Interactive analysis for 25+ <br/>GPS modeled tracks around the world.</h3>
    </div>
    <div className="apex__cta-dash">
      <DashGraph />
    </div>
    

  </div>
);

export default CTA;