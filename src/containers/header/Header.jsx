import React, { useEffect, useRef } from 'react';
import VanillaTilt from 'vanilla-tilt';

import './header.css';
// import people from '../../assets/people.png';
import track from '../../assets/track.png'

function Tilt(props) {
  const { options, ...rest } = props;
  const tilt = useRef(null);

  useEffect(() => {
    VanillaTilt.init(tilt.current, options);
  }, [options]);

  return <div ref={tilt} {...rest} />;
}

const options = {
  reverse:           false,  // reverse the tilt direction
  max:               20,     // max tilt rotation (degrees)
  perspective:       1000,   // Transform perspective, the lower the more extreme the tilt gets.
  scale:             1,      // 2 = 200%, 1.5 = 150%, etc..
  speed:             200,    // Speed of the enter/exit transition
  transition:        true,   // Set a transition on enter/exit.
  axis:              null,   // What axis should be disabled. Can be X or Y.
  reset:             true,   // If the tilt effect has to be reset on exit.
  easing:            "cubic-bezier(.03,.98,.52,.99)",    // Easing on enter/exit.
  glare:             true,   // if it should have a "glare" effect
  "max-glare":       0.6,      // the maximum "glare" opacity (1 = 100%, 0.5 = 50%)
  "glare-prerender": false   // false = VanillaTilt creates the glare elements for you, otherwise
                             // you need to add .js-tilt-glare>.js-tilt-glare-inner by yourself
};

const Header = () => {
  return (
    <div className='apex__header section__padding' id='home'>
      <div className='apex__header-content'>
        <h1 className='gradient__text'>Performance Perfected.  <br/>AI dynamics and driving analysis.</h1>
        <p>"If you’re going to push a piece of machinery to the limit, and expect to hold it together, you have to have 
          some sense of where that limit is. Look out there. Out there is the perfect lap. No mistakes, every gear change, every corner. Perfect."  
          <br/><br/>- Ken Miles, Ford v. Ferrari</p>
        <div className='apex__header-content__input'>
          <input type='email' placeholder='Your email adress'/>
          <button type="button">Get Started</button>
        </div>
      </div>
      
      <div className='apex__header-image'>  
        <Tilt options={options}>
          <div className='tiltcard'>
            <img src={track} alt='track'/>
          </div> 
        </Tilt>
      </div>
    </div>
  )
}

export default Header;