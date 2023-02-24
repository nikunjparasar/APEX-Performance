import React, { useState } from 'react';
import './navbar.css';
import {RiMenu3Line, RiCloseLine} from 'react-icons/ri';
import logo from '../../assets/logo.svg';

// menu funtional component
const Menu = () => (
  <>
  <p><a href='#home'>Home</a></p>
  <p><a href='#whatAPEX'>What is Apex</a></p>
  <p><a href='#possibility'>Test RS</a></p>
  <p><a href='#features'>Case Studies</a></p>
  <p><a href='#blog'>Library</a></p>
  </>
)

const Navbar = () => {
  const [toggleMenu, setToggleMenu] = useState(false);
  return (
    <div className='apex__navbar'>
      
      <div className='apex__navbar-links_logo'>
          <img src= {logo} alt="logo" />
      </div>

      <div className = 'apex__navbar-links'>
        <div className='apex__navbar-links_container'>
          <Menu />
        </div>

        <div className='apex__navbar-sign'>
          <p>Sign in</p>
          <button type='button'>Sign Up</button>
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
                  <p>Sign in</p>
                  <button type='button'>Sign Up</button>
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