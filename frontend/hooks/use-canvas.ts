"use client"

import { useRef, useState, useEffect, useCallback } from "react"

export function useCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isDrawing, setIsDrawing] = useState(false)

  const initCanvas = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    ctx.fillStyle = "white"
    ctx.fillRect(0, 0, canvas.width, canvas.height)
  }, [])

  useEffect(() => {
    initCanvas()
  }, [initCanvas])

  const getCoordinates = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return { x: 0, y: 0 }
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    return { 
      x: (e.clientX - rect.left) * scaleX, 
      y: (e.clientY - rect.top) * scaleY 
    }
  }

  const startDrawing = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDrawing(true)
    const { x, y } = getCoordinates(e)
    const ctx = canvasRef.current?.getContext("2d")
    if (!ctx) return
    ctx.beginPath()
    ctx.moveTo(x, y)
    ctx.lineWidth = 15
    ctx.lineCap = "round"
    ctx.strokeStyle = "black"
  }, [])

  const draw = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return
    const { x, y } = getCoordinates(e)
    const ctx = canvasRef.current?.getContext("2d")
    if (!ctx) return
    ctx.lineTo(x, y)
    ctx.stroke()
  }, [isDrawing])

  const stopDrawing = useCallback(() => {
    setIsDrawing(false)
    canvasRef.current?.getContext("2d")?.beginPath()
  }, [])

  const getPixelData = useCallback((): number[] => {
    const canvas = canvasRef.current
    if (!canvas) return []
    const ctx = canvas.getContext("2d")
    if (!ctx) return []

    // 1. Get Image Data and Bounding Box
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
    let minX = canvas.width, minY = canvas.height, maxX = 0, maxY = 0
    
    for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
            const i = (y * canvas.width + x) * 4
            // Check for non-white pixel (drawing is black)
            // Brightness check: if < 250 (not pure white), consider it drawn
            const brightness = (imageData.data[i] + imageData.data[i+1] + imageData.data[i+2]) / 3
            if (brightness < 250) {
                minX = Math.min(minX, x)
                minY = Math.min(minY, y)
                maxX = Math.max(maxX, x)
                maxY = Math.max(maxY, y)
            }
        }
    }

    if (minX > maxX) return [] // Empty canvas

    // 2. Crop and Square with Padding
    const padding = 20
    minX = Math.max(0, minX - padding)
    minY = Math.max(0, minY - padding)
    maxX = Math.min(canvas.width, maxX + padding)
    maxY = Math.min(canvas.height, maxY + padding)
    
    const w = maxX - minX
    const h = maxY - minY
    const size = Math.max(w, h)

    const cropCanvas = document.createElement("canvas")
    cropCanvas.width = size
    cropCanvas.height = size
    const cropCtx = cropCanvas.getContext("2d")
    if (!cropCtx) return []

    // Fill white first
    cropCtx.fillStyle = "white"
    cropCtx.fillRect(0, 0, size, size)
    // Draw image centered
    cropCtx.drawImage(canvas, minX, minY, w, h, (size-w)/2, (size-h)/2, w, h)

    // 3. Resize to 28x28
    const tempCanvas = document.createElement("canvas")
    tempCanvas.width = 28
    tempCanvas.height = 28
    const tempCtx = tempCanvas.getContext("2d")
    if (!tempCtx) return []

    tempCtx.imageSmoothingEnabled = true
    tempCtx.imageSmoothingQuality = "high"
    tempCtx.drawImage(cropCanvas, 0, 0, 28, 28)

    // 4. Extract and Normalize Pixels
    const finalData = tempCtx.getImageData(0, 0, 28, 28)
    const pixels: number[] = []
    
    for (let i = 0; i < finalData.data.length; i += 4) {
        // Invert: 0 is white, 1 is black
        // Normalize: 0.0 to 1.0
        const val = (255 - finalData.data[i]) / 255.0
        pixels.push(val)
    }

    return pixels
  }, [])

  return {
    canvasRef,
    initCanvas,
    startDrawing,
    draw,
    stopDrawing,
    getPixelData
  }
}
