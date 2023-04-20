import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './cpprunner.css';


function CppRunner() {
  const [output, setOutput] = useState('');

  useEffect(() => {
    // Make a GET request to the server endpoint to get the contents of output.txt
    axios.get('/output.txt')
      .then(response => {
        setOutput(response.data);
      })
      .catch(error => {
        console.error(error);
      });
  }, []);

  return (
    <div>
      <h2>Control Parameters: Formula 1</h2>
      <pre>{output}</pre>
      <p><br/></p>

    </div>
  );
}

export default CppRunner;