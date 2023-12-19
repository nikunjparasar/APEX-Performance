import React from 'react';
import DashGraph from './DashGraph';
import CppRunner from './cpprunner';

const MyGrid = () => {
  return (
    <div className="grid-container">
      <div className="grid-item">
        <DashGraph />
      </div>
      <div className="grid-item">
        <DashGraph />
      </div>
      <div className="grid-item">
        <DashGraph />
      </div>
      <div className="grid-item">
        <CppRunner filePath={'output.txt'} />
      </div>
      <div className="grid-item">
        <DashGraph />
      </div>
      <div className="grid-item">
        <DashGraph />
      </div>
    </div>
  );
};

export default MyGrid;
