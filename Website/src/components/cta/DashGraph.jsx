import React from 'react';

const DashGraph = () => {
  return (
    <iframe
      title="Dash Graph"
      src="http://localhost:8051" // or your deployed Dash app URL
      width="100%"
      height="500px"
      frameBorder="0"
    />
  );
};

export default DashGraph;