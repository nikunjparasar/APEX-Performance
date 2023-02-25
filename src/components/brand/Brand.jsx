import React from 'react';
import './brand.css';
import {f1, nvidia, google, tensorflow, matlab} from './imports';

const Brand = () => {
  return (
    <div className='apex__brand section__padding'>
      <div className='tensorflow'>
        <img src= {tensorflow} alt= 'tensorflow'/>
      </div>

      <div className='matlab'>
        <img src= {matlab} alt= 'matlab'/>
      </div>

      <div className='f1'>
        <img src= {f1} alt= 'f1'/>
      </div>

      <div className = 'nvidia'>
        <img src= {nvidia} alt= 'nvidia'/>
      </div>

      <div>
        <img src= {google} alt= 'google'/>
      </div>

    </div>
  )
}

export default Brand;