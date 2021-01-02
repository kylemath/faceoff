import React, { useRef } from 'react'

//Canvas Stuff
//https://medium.com/@pdx.lucasm/canvas-with-react-js-32e133c05258
const Canvas = props => {

  const { draw, canvas, ...rest } = props
  const canvasRef = useRef(null)

  const styles = {	width:'50%'  }
  return (
  	<canvas style={styles} id={canvas} ref={canvasRef} {...rest}/>
  )
}

export default Canvas
