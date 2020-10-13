// OpenCV type definitions required by worker.ts

type NativeObject = {
  delete: () => void
}

type Mat = {
  rows: number
  cols: number
  data: Uint8ClampedArray
  channels(): number
} & NativeObject

type Vector<T> = {
  push_back(e: T): void
  get: (i: number) => T
  size: () => number
} & NativeObject

type Point = {
  x: number
  y: number
}

type KeyPoint = {
  pt: Point
  size: number
  angle: number
  response: number
  octave: number
  class_id: number
}
type KeyPointVector = Vector<KeyPoint>

type AKAZE = {
  detectAndCompute(
    image: Mat,
    mask: Mat,
    keypoints: KeyPointVector,
    descriptors: Mat,
    useProvidedKeypoints: boolean
  ): void
} & NativeObject

type DMatch = {
  queryIdx: number
  trainIdx: number
  imgIdx: number
  distance: number
}
type DMatchVector = Vector<DMatch>
type DMatchVectorVector = Vector<DMatchVector>

type BFMatcher = {
  knnMatch(
    queryDescriptors: Mat,
    trainDescritprs: Mat,
    matches: DMatchVectorVector,
    k: number,
    mask: Mat,
    compactResult: boolean
  ): void
} & NativeObject

type OpenCVObject = {
  Mat: new () => Mat
  KeyPointVector: new () => KeyPointVector
  AKAZE: new () => AKAZE
  DMatchVectorVector: new () => DMatchVectorVector
  BFMatcher: new (normType: number, crossCheck: boolean) => BFMatcher
  DMatchVector: new () => DMatchVector

  COLOR_RGBA2RGB: number
  COLOR_RGB2RGBA: number

  matFromImageData: (imageData: ImageData) => Mat
  cvtColor: (src: Mat, dest: Mat, code: number, destChannel: number) => void
  drawMatchesKnn: (
    img1: Mat,
    kp1: KeyPointVector,
    img2: Mat,
    kp2: KeyPointVector,
    matches1to2: DMatchVectorVector, outputImage: Mat) => void
}