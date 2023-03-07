import React from 'react';

import apexLogo from '../../assets/jesko.png';
import './footer.css';

const Footer = () => (
  <div className="apex__footer ">
      <div className="apex__footer-heading">
        {/* <h1 className="gradient__text">Do you want to step in to the future before others</h1> */}
      </div>
      <div className="apex__footer-btn">
        <p>Future: View Github</p>
      </div>

      <></>





    <div className="apex__footer-links">
      <div className="apex__footer-links_logo">
        <img src={apexLogo} alt="apex_logo" />
        <p>APEX Performance Motorsports <br /> All Rights Reserved</p>
      </div>
      <div className="apex__footer-links_div">
        <h4>Links</h4>
        <p>References</p>
        <p>Social Media</p>
        <p>Counters</p>
        <p>Contact</p>
      </div>
      <div className="apex__footer-links_div">
        <h4>Company</h4>
        <p>Terms & Conditions </p>
        <p>Privacy Policy</p>
        <p>Contact</p>
      </div>
      <div className="apex__footer-links_div">
        <h4>Get in touch</h4>
        <p>Cupertino, CA</p>
        <p>@nikkparasar</p>
      </div>
    </div>

    <div className="apex__footer-copyright">
      <p>@2023 APEX. All rights reserved.</p>
    </div>
  </div>
);

export default Footer;