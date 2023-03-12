import React from 'react';

const DashGraph = () => {
  return (
    <iframe
      title="Dash Graph"
      src="http://localhost:8050" // or your deployed Dash app URL
      width="750px"
      height="770px"
      frameBorder="0"
    />
  );
};

export default DashGraph;