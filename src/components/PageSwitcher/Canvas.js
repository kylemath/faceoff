import React, { useRef } from 'react'

//Canvas Stuff
//https://medium.com/@pdx.lucasm/canvas-with-react-js-32e133c05258
const Canvas = props => {

  const { draw, canvas, ...rest } = props
  const canvasRef = useRef(null)

  return <canvas id={canvas} ref={canvasRef} {...rest}/>
}

export default Canvas
