importScripts('./opencv.js')

declare const cv: () => Promise<OpenCVObject>

cv().then(cv => {
  addEventListener('message', (e: MessageEvent): void => {
    const data = e.data as { type: string, target: ImageData, whole: ImageData }
    switch (data.type) {
      case 'match':
        match(cv, data.target, data.whole)
        break
    }
  })
  postMessage({
    type: 'initialized'
  })
  console.log('worker is initialized.')
})

function match(cv: OpenCVObject, target: ImageData, whole: ImageData): void {
  const rawTarget = cv.matFromImageData(target)
  const targetImage = new cv.Mat()
  cv.cvtColor(rawTarget, targetImage, cv.COLOR_RGBA2RGB, 0)
  const rawWhole = cv.matFromImageData(whole)
  const wholeImage = new cv.Mat()
  cv.cvtColor(rawWhole, wholeImage, cv.COLOR_RGBA2RGB, 0)

  const mask = new cv.Mat()
  const kp1 = new cv.KeyPointVector()
  const des1 = new cv.Mat()
  const kp2 = new cv.KeyPointVector()
  const des2 = new cv.Mat()
  const akaze = new cv.AKAZE()
  akaze.detectAndCompute(targetImage, mask, kp1, des1, false)
  akaze.detectAndCompute(wholeImage, mask, kp2, des2, false)

  const matches = new cv.DMatchVectorVector()
  const bfMatcher = new cv.BFMatcher(2, false)
  bfMatcher.knnMatch(des1, des2, matches, 2, mask, false)
  const ratio = 0.5
  const good = new cv.DMatchVectorVector()
  for (let i = 0; i < matches.size(); i++) {
    const match = matches.get(i)
    const m1 = match.get(0)
    const m2 = match.get(1)
    if (m1.distance < ratio * m2.distance) {
      const t = new cv.DMatchVector()
      t.push_back(m1)
      good.push_back(t)
    }
  }
  const matchingImage = new cv.Mat()
  cv.drawMatchesKnn(targetImage, kp1, wholeImage, kp2, good, matchingImage)

  const result = new cv.Mat()
  cv.cvtColor(matchingImage, result, cv.COLOR_RGB2RGBA, 0)
  const type = 'response'
  const imageData = {
    data: new Uint8ClampedArray(result.data),
    width: matchingImage.cols,
    height: matchingImage.rows
  }
  const channels = matchingImage.channels()
  postMessage({ type, imageData, channels }, [imageData.data.buffer]);

  [
    rawTarget,
    targetImage,
    rawWhole,
    wholeImage,
    akaze,
    mask,
    kp1,
    des1,
    kp2,
    des2,
    bfMatcher,
    matches,
    good,
    matchingImage,
    result
  ].forEach(obj => obj.delete())
}
