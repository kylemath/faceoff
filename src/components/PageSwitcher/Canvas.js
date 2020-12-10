import React, { useRef, useEffect } from 'react'

//Canvas Stuff
//https://medium.com/@pdx.lucasm/canvas-with-react-js-32e133c05258
const Canvas = props => {

  const { draw, ...rest } = props
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const context = canvas.getContext('2d')
    let frameCount = 0
    let animationFrameId

    const render = () => {
      frameCount++
      draw(canvas, context, frameCount)
      animationFrameId = window.requestAnimationFrame(render)
    }
    render()

    return () => {
      window.cancelAnimationFrame(animationFrameId)
    }
  }, [draw])

  return <canvas ref={canvasRef} {...rest}/>
}

export default Canvas
