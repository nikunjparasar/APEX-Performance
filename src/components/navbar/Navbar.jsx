import React, { useState } from 'react';
import './navbar.css';
import {RiMenu3Line, RiCloseLine} from 'react-icons/ri';
// import logo from '../../assets/logo.svg';
import jesko from '../../assets/jesko.png';

// menu funtional component
const Menu = () => (
  <>
  <p><a href='#whatAPEX'>ABOUT</a></p>
  <p><a href='#possibility'>RS</a></p>
  <p><a href='#features'>AI</a></p>
  <p><a href='#blog'>REFERENCES</a></p>
  </>
)

const Navbar = () => {
  const [toggleMenu, setToggleMenu] = useState(false);
  return (
    <div className='apex__navbar'>
      
      <div className='apex__navbar-links_logo'>
          <p><a href='#home'>APEX</a></p>
          <a href = '#test'><img src= {jesko} alt="jesko" /></a>
          
      </div>

      <div className = 'apex__navbar-links'>
        <div className='apex__navbar-links_container'>
          <Menu />
        </div>

        <div className='apex__navbar-sign'>
          <button type='button'>TEST</button>
        </div>

        <div className='apex__navbar-menu'>
          {toggleMenu
            ? <RiCloseLine color="#FFF" size={27} onClick={() => setToggleMenu(false)} />
            : <RiMenu3Line color="#FFF" size={27} onClick={() => setToggleMenu(true)} />
          }
          {/* render only if menu toggled */}
          {toggleMenu && (
            <div className='apex__navbar-menu_container scale-up-center'>
              <div className='apex__navbar-menu_container-links'>
                <Menu />
                <div className='apex__navbar-menu_container-links-sign'>
                  <button type='button'>TEST</button>
                </div>
              </div>
            </div>
          )
          }
        </div>
      </div>
    </div>
  )
}

export default Navbar;