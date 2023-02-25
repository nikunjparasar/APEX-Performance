import React from 'react';
import './header.css';
// import people from '../../assets/people.png';
import track from '../../assets/track.png'
import tilt from './vanilla-tilt.js';

const Header = () => {
  return (
    <div className='apex__header section__padding' id='home'>
      <div className='apex__header-content'>
        <h1 className='gradient__text'>Performance Perfected.  <br/>AI vehicle dynamics and driving analysis.</h1>
        <p>"If you’re going to push a piece of machinery to the limit, and expect to hold it together, you have to have 
          some sense of where that limit is. Look out there. Out there is the perfect lap. No mistakes, every gear change, every corner. Perfect."  
          <br/><br/>- Ken Miles, Ford v. Ferrari</p>
        <div className='apex__header-content__input'>
          <input type='email' placeholder='Your email adress'/>
          <button type="button">Get Started</button>
        </div>

        {/* <div className='apex__header-content__people'>
          <img src={people} alt = 'people'/>
          <p>1600 people requested access</p>
        </div> */}
      </div>
      <div className='apex__header-image'>
        <script src={tilt}></script>
        <div className='tiltcard' data-tilt data-tilt-glare>
          <img src={track} alt='track'/>
        </div>
        
       
      </div>
    </div>
  )
}

export default Header;