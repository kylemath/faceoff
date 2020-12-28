import React, { useRef } from 'react'

//Canvas Stuff
//https://medium.com/@pdx.lucasm/canvas-with-react-js-32e133c05258
const OtherCanvas = props => {

  const { draw, ...rest } = props
  const canvasRef = useRef(null)

  return <canvas id="other_canvas" ref={canvasRef} {...rest}/>
}

export default OtherCanvas
