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

function searchFarthestPoints(
  target: ImageData,
  whole: ImageData,
  kp1: KeyPointVector,
  kp2: KeyPointVector,
  good: DMatchVectorVector
): { p1: Point, p2: Point }[] {
  if (good.size() <= 1) {
    throw new Error('Not enough good points.')
  }

  const width = Math.max(target.width, whole.height)
  const height = Math.max(target.height, whole.height)
  const tlbr: { p: Point, qi: number, ti: number }[] = [
    { p: { x: width, y: height }, qi: 0, ti: 0 }, // top
    { p: { x: width, y: height }, qi: 0, ti: 0 }, // left
    { p: { x: 0, y: 0 }, qi: 0, ti: 0 }, // bottom
    { p: { x: 0, y: 0 }, qi: 0, ti: 0 }, // right
  ]
  for (let i = 0; i < good.size(); i++) {
    const m = good.get(i).get(0)
    const m2 = kp2.get(m.trainIdx)
    let updateTlbrIndex: number = -1
    if (m2.pt.y < tlbr[0].p.y) {
      updateTlbrIndex = 0
    }
    if (m2.pt.x < tlbr[1].p.x) {
      updateTlbrIndex = 1
    }
    if (m2.pt.y > tlbr[2].p.y) {
      updateTlbrIndex = 2
    }
    if (m2.pt.x > tlbr[3].p.x) {
      updateTlbrIndex = 3
    }
    if (updateTlbrIndex >= 0) {
      const t = tlbr[updateTlbrIndex]
      t.p = m2.pt
      t.qi = m.queryIdx
      t.ti = m.trainIdx
    }
  }
  let maxSquaredDistance = 0
  let maxSdIndexPair: { i1: number, i2: number } = { i1: 0, i2: 0 }
  for (let i = 0; i < tlbr.length; i++) {
    for (let ii = i + 1; ii < tlbr.length; ii++) {
      const p1 = tlbr[i].p
      const p2 = tlbr[ii].p
      const sd = (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2
      if (sd > maxSquaredDistance) {
        maxSquaredDistance = sd
        maxSdIndexPair = { i1: i, i2: ii }
      }
    }
  }
  return [
    {
      p1: kp1.get(tlbr[maxSdIndexPair.i1].qi).pt,
      p2: kp1.get(tlbr[maxSdIndexPair.i2].qi).pt
    },
    {
      p1: kp2.get(tlbr[maxSdIndexPair.i1].ti).pt,
      p2: kp2.get(tlbr[maxSdIndexPair.i2].ti).pt
    },
  ]
}

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

  const ps = searchFarthestPoints(target, whole, kp1, kp2, good)
  console.log('The farthest points of target', ps[0])
  console.log('The farthest points of whole', ps[1])

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
