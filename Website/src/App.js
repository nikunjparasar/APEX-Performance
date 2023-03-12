import React from 'react';

import {Footer, Possibility, Features, Header} from './containers';
import {CTA, Brand, Navbar} from './components';
import './App.css';
const App = () => {
  return (
    <div className='App'>
      <div class="container">
            <div class="text-wrapper">
                <div class="text-1 text">APEX PERFORMANCE</div>
                <div class="text-2 text">APEX PERFORMANCE</div>
                <div class="text-3 text">APEX PERFORMANCE</div>
                <div class="text-4 text">APEX PERFORMANCE</div>
                <div class="text-5 text">APEX PERFORMANCE</div>
                <div class="text-6 text">APEX PERFORMANCE</div>
                <div class="text-7 text">APEX PERFORMANCE</div>
                <div class="text-8 text">APEX PERFORMANCE</div>
                <div class="text-9 text">APEX PERFORMANCE</div>
                <div class="text-10 text">APEX PERFORMANCE</div>
                <div class="text-11 text">APEX PERFORMANCE</div>
            </div>
        </div>

        <div className='gradient__bg'>
          <Navbar />
          <Header />
        </div>
        <Brand />
        <Features />
        <Possibility />
        <CTA />
        <Footer />
    </div>
    
  )
}

export default App;