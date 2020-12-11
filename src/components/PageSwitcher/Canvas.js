import React, { useRef } from 'react'

//Canvas Stuff
//https://medium.com/@pdx.lucasm/canvas-with-react-js-32e133c05258
const Canvas = props => {

  const { draw, ...rest } = props
  const canvasRef = useRef(null)

  return <canvas id="the_canvas" ref={canvasRef} {...rest}/>
}

export default Canvas
