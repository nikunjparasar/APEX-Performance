import React from 'react';

import apexLogo from '../../assets/jesko.png';
import './footer.css';

const Footer = () => (
  <div className="apex__footer ">
      <div className="apex__footer-heading">
        {/* <h1 className="gradient__text">Do you want to step in to the future before others</h1> */}
      </div>
      <div className="apex__footer-btn">
        <p>View Github</p>
      </div>

      <></>





    <div className="apex__footer-links">
      <div className="apex__footer-links_logo">
        <img src={apexLogo} alt="apex_logo" />
        <p>Crechterwoord K12 182 DK Alknjkcb, <br /> All Rights Reserved</p>
      </div>
      <div className="apex__footer-links_div">
        <h4>Links</h4>
        <p>Overons</p>
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
        <p>Crechterwoord K12 182 DK Alknjkcb</p>
        <p>085-132567</p>
        <p>info@payme.net</p>
      </div>
    </div>

    <div className="apex__footer-copyright">
      <p>@2023 APEX. All rights reserved.</p>
    </div>
  </div>
);

export default Footer;