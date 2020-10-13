function loadImage(path: string, canvasId: string): Promise<void> {
  return new Promise(resolve => {
    const image = new Image()
    image.addEventListener('load', () => {
      const canvas = document.getElementById(canvasId) as HTMLCanvasElement
      if (!canvas) {
        throw new Error(`No canvas ${canvasId}.`)
      }
      canvas.width = image.width
      canvas.height = image.height
      const context = canvas.getContext('2d')
      if (!context) {
        throw new Error(`No context 2d in ${canvasId}.`)
      }
      context.drawImage(image, 0, 0)
      resolve()
    })
    image.src = path
  })
}

function showImage({ data, height, width }: ImageData, canvasId: string): void {
  const canvas = document.getElementById(canvasId) as HTMLCanvasElement
  if (!canvas) {
    throw new Error(`No canvas ${canvasId}.`)
  }
  const context = canvas.getContext('2d')
  if (!context) {
    throw new Error(`No context 2d in ${canvasId}.`)
  }
  context.clearRect(0, 0, canvas.width, canvas.height)
  canvas.width = width
  canvas.height = height
  context.putImageData(new ImageData(data, width, height), 0, 0)
}

function getImageData(canvasId: string): ImageData {
  const canvas = document.getElementById(canvasId) as HTMLCanvasElement
  if (!canvas) {
    throw new Error(`No canvas ${canvasId}.`)
  }
  const context = canvas.getContext('2d')
  if (!context) {
    throw new Error(`No context 2d of ${canvasId}.`)
  }
  return context.getImageData(0, 0, canvas.width, canvas.height)
}

async function main() {
  const matchButton = document.getElementById('match') as HTMLButtonElement
  if (!matchButton) {
    throw new Error('No match buttom.')
  }
  matchButton.disabled = true
  matchButton.textContent = 'Loading...'
  matchButton.addEventListener('click', () => {
    const target = getImageData('target')
    const whole = getImageData('whole')
    worker.postMessage({
      type: 'match',
      target,
      whole
    }, [target.data.buffer, whole.data.buffer])
  })

  await Promise.all([
    loadImage('./target.png', 'target'),
    loadImage('./whole.png', 'whole')
  ])
  const worker = new Worker('./worker.js')
  worker.addEventListener('message', (ev: MessageEvent): void => {
    const data = ev.data as { type: 'initialized' | 'response', imageData: ImageData }
    switch (data.type) {
      case 'initialized':
        matchButton.disabled = false
        matchButton.textContent = 'Match'
        console.log('initialized')
        break
      case 'response':
        showImage(data.imageData, 'result')
        break;
    }
  })
}

main()
