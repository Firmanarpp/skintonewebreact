import { useEffect, useMemo, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import * as faceDetection from '@tensorflow-models/face-detection'
import { createClient } from '@supabase/supabase-js'

const MST_COLORS = [
  '#f6ede4',
  '#f3e7db',
  '#f7ead0',
  '#eadaba',
  '#d7bd96',
  '#a07e56',
  '#825c43',
  '#604134',
  '#3a312a',
  '#292420',
]

const CLASS_LABELS = [
  'MST1',
  'MST2',
  'MST3',
  'MST4',
  'MST5',
  'MST6',
  'MST7',
  'MST8',
  'MST9',
  'MST10',
]

const CLOTHING_RECOMMENDATIONS = {
  light: {
    recommended: [
      { name: 'Navy Blue', hex: '#000080' },
      { name: 'Royal Purple', hex: '#7851a9' },
      { name: 'Emerald Green', hex: '#046307' },
      { name: 'Burgundy', hex: '#800020' },
      { name: 'Sapphire Blue', hex: '#0f52ba' },
    ],
    avoid: [
      { name: 'Orange', hex: '#ffa500' },
      { name: 'Bright Yellow', hex: '#ffff00' },
      { name: 'Pastel Colors', hex: '#fadadd' },
    ],
  },
  'light medium': {
    recommended: [
      { name: 'Teal', hex: '#008080' },
      { name: 'Cobalt Blue', hex: '#0047ab' },
      { name: 'Lavender', hex: '#e6e6fa' },
      { name: 'Ruby Red', hex: '#9b111e' },
      { name: 'Forest Green', hex: '#228b22' },
    ],
    avoid: [
      { name: 'Brown', hex: '#5c4033' },
      { name: 'Khaki', hex: '#c3b091' },
      { name: 'Olive', hex: '#808000' },
    ],
  },
  medium: {
    recommended: [
      { name: 'Coral', hex: '#ff7f50' },
      { name: 'Turquoise', hex: '#40e0d0' },
      { name: 'Olive Green', hex: '#556b2f' },
      { name: 'Royal Blue', hex: '#4169e1' },
      { name: 'Magenta', hex: '#c71585' },
    ],
    avoid: [
      { name: 'Neon Colors', hex: '#39ff14' },
      { name: 'White', hex: '#ffffff' },
      { name: 'Black', hex: '#000000' },
    ],
  },
  'medium deep': {
    recommended: [
      { name: 'Gold', hex: '#ffd700' },
      { name: 'Mustard Yellow', hex: '#ffdb58' },
      { name: 'Orange', hex: '#ffa500' },
      { name: 'Kelly Green', hex: '#4cbb17' },
      { name: 'Electric Blue', hex: '#7df9ff' },
    ],
    avoid: [
      { name: 'Pastel Colors', hex: '#fadadd' },
      { name: 'Beige', hex: '#f5f5dc' },
      { name: 'Silver', hex: '#c0c0c0' },
    ],
  },
  deep: {
    recommended: [
      { name: 'Bright Yellow', hex: '#ffff00' },
      { name: 'Fuchsia', hex: '#ff00ff' },
      { name: 'Lime Green', hex: '#32cd32' },
      { name: 'Bright Orange', hex: '#ff4500' },
      { name: 'Aqua', hex: '#00ffff' },
    ],
    avoid: [
      { name: 'Dark Colors', hex: '#2f4f4f' },
      { name: 'Brown', hex: '#5c4033' },
      { name: 'Navy', hex: '#000080' },
    ],
  },
}

const MODEL_URL = '/models/mobilenetv2_mst_model94/model.json'

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY
const SUPABASE_BUCKET = import.meta.env.VITE_SUPABASE_BUCKET

const supabaseClient = SUPABASE_URL && SUPABASE_ANON_KEY ? createClient(SUPABASE_URL, SUPABASE_ANON_KEY) : null
const backendReady = tf.setBackend('cpu').then(() => tf.ready())

function App() {
  const [navOpen, setNavOpen] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [processedPreviewUrl, setProcessedPreviewUrl] = useState('')
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')
  const [activeTab, setActiveTab] = useState('recommended')
  const [accordionOpen, setAccordionOpen] = useState([false, false, false, false])
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef(null)
  const modelRef = useRef(null)
  const faceDetectorRef = useRef(null)

  useEffect(() => {
    ensureBackend()
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }
      if (processedPreviewUrl) {
        URL.revokeObjectURL(processedPreviewUrl)
      }
    }
  }, [previewUrl, processedPreviewUrl])

  const mstScaleItems = useMemo(
    () =>
      MST_COLORS.map((color, index) => ({
        color,
        label: index + 1,
        mst: `MST${index + 1}`,
      })),
    []
  )

  const handleSelectFile = (file) => {
    if (!file) {
      return
    }
    if (!file.type.startsWith('image/')) {
      window.alert('Please select a valid image file.')
      return
    }
    if (file.size > 5 * 1024 * 1024) {
      window.alert('Image size exceeds 5MB limit.')
      return
    }
    setSelectedFile(file)
    setResults(null)
    setErrorMessage('')
    setActiveTab('recommended')
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }
    const nextPreviewUrl = URL.createObjectURL(file)
    setPreviewUrl(nextPreviewUrl)
  }

  const handleFileChange = (event) => {
    const file = event.target.files?.[0]
    handleSelectFile(file)
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setDragActive(false)
    const file = event.dataTransfer.files?.[0]
    handleSelectFile(file)
  }

  const handleAnalyze = async () => {
    if (!selectedFile) {
      return
    }
    setLoading(true)
    setErrorMessage('')
    try {
      await ensureBackend()
      const model = await loadModel()
      const timestamp = Date.now()
      const { tensorInput, processedCanvas, luminance, faceDetected } = await createTensorFromFile(selectedFile)
      const predictionTensor = model.predict(tensorInput)
      const predictionData = await predictionTensor.data()
      predictionTensor.dispose()
      tensorInput.dispose()
      const { predictedLabel, confidence } = getPrediction(predictionData)
      const adjustedLabel = adjustPredictionWithLuminance(predictedLabel, luminance)
      const mstIndex = parseInt(adjustedLabel.replace('MST', ''), 10) - 1
      const skinToneGroup = getSkinToneGroup(adjustedLabel)
      const recommendations = CLOTHING_RECOMMENDATIONS[skinToneGroup]
      const processedBlob = await canvasToBlob(processedCanvas)
      const processedLocalUrl = URL.createObjectURL(processedBlob)
      if (processedPreviewUrl) {
        URL.revokeObjectURL(processedPreviewUrl)
      }
      setProcessedPreviewUrl(processedLocalUrl)
      const uploadBase = `uploads/${timestamp}_${selectedFile.name}`
      const processedBase = `processed/${timestamp}_${selectedFile.name.replace(/\.[^/.]+$/, '')}.jpg`
      const imageUrl = await uploadToSupabase(selectedFile, uploadBase, selectedFile.type)
      const processedUrl = await uploadToSupabase(processedBlob, processedBase, 'image/jpeg')
      setResults({
        prediction: adjustedLabel,
        confidence: confidence * 100,
        mstColor: MST_COLORS[mstIndex],
        mstIndex,
        imageUrl: imageUrl || previewUrl,
        processedImageUrl: processedUrl || processedLocalUrl,
        faceDetected,
        skinToneGroup,
        recommendations,
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setErrorMessage(message)
      window.alert(`An error occurred during analysis: ${message}`)
    } finally {
      setLoading(false)
    }
  }

  const ensureBackend = async () => {
    await backendReady
  }

  const loadModel = async () => {
    if (modelRef.current) {
      return modelRef.current
    }
    await ensureBackend()
    modelRef.current = await tf.loadLayersModel(MODEL_URL)
    return modelRef.current
  }

  const loadFaceDetector = async () => {
    if (faceDetectorRef.current) {
      return faceDetectorRef.current
    }
    await ensureBackend()
    const model = faceDetection.SupportedModels.MediaPipeFaceDetector
    const detector = await faceDetection.createDetector(model, {
      runtime: 'tfjs',
      modelType: 'short',
    })
    faceDetectorRef.current = detector
    return detector
  }

  const loadImageFromFile = (file) =>
    new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file)
      const image = new Image()
      image.onload = () => {
        URL.revokeObjectURL(url)
        resolve(image)
      }
      image.onerror = () => {
        URL.revokeObjectURL(url)
        reject(new Error('Failed to load image file.'))
      }
      image.src = url
    })

  const createTensorFromFile = async (file) => {
    let imageBitmap = null
    let imageElement = null
    try {
      imageBitmap = await createImageBitmap(file)
    } catch {
      imageElement = await loadImageFromFile(file)
    }
    const sourceWidth = imageBitmap ? imageBitmap.width : imageElement.width
    const sourceHeight = imageBitmap ? imageBitmap.height : imageElement.height
    const size = 224
    const sourceCanvas = document.createElement('canvas')
    sourceCanvas.width = sourceWidth
    sourceCanvas.height = sourceHeight
    const sourceCtx = sourceCanvas.getContext('2d')
    if (imageBitmap) {
      sourceCtx.drawImage(imageBitmap, 0, 0)
    } else {
      sourceCtx.drawImage(imageElement, 0, 0)
    }
    let faceDetected = false
    let cropArea = {
      x: 0,
      y: 0,
      width: imageBitmap.width,
      height: imageBitmap.height,
    }
    try {
      const detector = await loadFaceDetector()
      const faces = await detector.estimateFaces(sourceCanvas, { flipHorizontal: false })
      if (faces && faces.length > 0) {
        const box = faces[0].box ?? faces[0].boundingBox
        if (box) {
          const xMin = box.xMin ?? box.x ?? 0
          const yMin = box.yMin ?? box.y ?? 0
          const width = box.width ?? Math.max(0, (box.xMax ?? 0) - (box.xMin ?? 0))
          const height = box.height ?? Math.max(0, (box.yMax ?? 0) - (box.yMin ?? 0))
          const marginX = width * 0.2
          const marginY = height * 0.2
          const cropX = Math.max(0, xMin - marginX)
          const cropY = Math.max(0, yMin - marginY)
          const cropW = Math.min(sourceWidth - cropX, width + marginX * 2)
          const cropH = Math.min(sourceHeight - cropY, height + marginY * 2)
          if (cropW > 0 && cropH > 0) {
            cropArea = {
              x: cropX,
              y: cropY,
              width: cropW,
              height: cropH,
            }
            faceDetected = true
          }
        }
      }
    } catch {
      faceDetected = false
    }
    const canvas = document.createElement('canvas')
    canvas.width = size
    canvas.height = size
    const ctx = canvas.getContext('2d')
    ctx.drawImage(
      sourceCanvas,
      cropArea.x,
      cropArea.y,
      cropArea.width,
      cropArea.height,
      0,
      0,
      size,
      size
    )
    const imageData = ctx.getImageData(0, 0, size, size)
    const luminance = calculateLuminance(imageData.data)
    const tensorInput = tf.tidy(() =>
      tf.browser.fromPixels(canvas).toFloat().div(255).expandDims(0)
    )
    return { tensorInput, processedCanvas: canvas, luminance, faceDetected }
  }

  const calculateLuminance = (data) => {
    let total = 0
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i]
      const g = data[i + 1]
      const b = data[i + 2]
      total += 0.299 * r + 0.587 * g + 0.114 * b
    }
    return total / (data.length / 4)
  }

  const getPrediction = (predictionData) => {
    let maxValue = -Infinity
    let maxIndex = 0
    predictionData.forEach((value, index) => {
      if (value > maxValue) {
        maxValue = value
        maxIndex = index
      }
    })
    return {
      predictedLabel: CLASS_LABELS[maxIndex] || 'MST5',
      confidence: maxValue,
    }
  }

  const adjustPredictionWithLuminance = (predictedLabel, luminance) => {
    let mstNumber = parseInt(predictedLabel.replace('MST', ''), 10)
    if (luminance < 80 && mstNumber >= 4 && mstNumber <= 7) {
      mstNumber = Math.min(10, mstNumber + 2)
    }
    if (luminance < 50 && mstNumber < 8) {
      mstNumber = Math.min(10, mstNumber + 3)
    }
    return `MST${mstNumber}`
  }

  const getSkinToneGroup = (label) => {
    const mstNumber = parseInt(label.replace('MST', ''), 10)
    if (mstNumber <= 2) {
      return 'light'
    }
    if (mstNumber <= 4) {
      return 'light medium'
    }
    if (mstNumber <= 6) {
      return 'medium'
    }
    if (mstNumber <= 8) {
      return 'medium deep'
    }
    return 'deep'
  }

  const canvasToBlob = (canvas) =>
    new Promise((resolve) => {
      canvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.9)
    })

  const uploadToSupabase = async (file, path, contentType) => {
    if (!supabaseClient || !SUPABASE_BUCKET) {
      return null
    }
    const { error } = await supabaseClient.storage.from(SUPABASE_BUCKET).upload(path, file, {
      contentType,
      upsert: true,
    })
    if (error) {
      return null
    }
    const { data } = supabaseClient.storage.from(SUPABASE_BUCKET).getPublicUrl(path)
    return data?.publicUrl ?? null
  }

  const toggleAccordion = (index) => {
    setAccordionOpen((prev) =>
      prev.map((value, idx) => (idx === index ? !value : value))
    )
  }

  return (
    <div>
      <nav className="navbar">
        <div className="nav-container">
          <a href="#" className="logo">
            <i className="fas fa-palette"></i> SkinTone<span>AI</span>
          </a>
          <ul className={`nav-links ${navOpen ? 'active' : ''}`}>
            {['about', 'analyzer', 'how-it-works', 'faq'].map((section) => (
              <li key={section}>
                <a href={`#${section}`} onClick={() => setNavOpen(false)}>
                  {section.replace('-', ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                </a>
              </li>
            ))}
          </ul>
          <div className={`hamburger ${navOpen ? 'active' : ''}`} onClick={() => setNavOpen((prev) => !prev)}>
            <span className="bar"></span>
            <span className="bar"></span>
            <span className="bar"></span>
          </div>
        </div>
      </nav>

      <section className="hero">
        <div className="hero-container">
          <div className="hero-content">
            <h1>Discover Your Perfect Color Palette</h1>
            <p>Upload a photo and our AI will analyze your skin tone to provide personalized clothing color recommendations that complement your natural beauty.</p>
            <a href="#analyzer" className="hero-btn" id="try-it-now-btn">Try It Now <i className="fas fa-arrow-right"></i></a>
          </div>
          <div className="hero-image">
            <img src="/images/hero-image.svg" alt="Skin Tone Analysis" />
          </div>
        </div>
        <div className="wave-separator">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 320">
            <path fill="#ffffff" fillOpacity="1" d="M0,288L48,272C96,256,192,224,288,197.3C384,171,480,149,576,165.3C672,181,768,235,864,250.7C960,267,1056,245,1152,208C1248,171,1344,117,1392,90.7L1440,64L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"></path>
          </svg>
        </div>
      </section>

      <section id="about" className="about">
        <div className="container">
          <div className="section-header">
            <h2>Why Skin Tone Matters</h2>
            <p>Understanding your skin tone helps you make better fashion choices and enhances your personal style</p>
          </div>

          <div className="features">
            {[
              { icon: 'tshirt', title: 'Better Clothing Choices', text: 'Identify colors that naturally complement your skin tone for a more flattering look.' },
              { icon: 'paint-brush', title: 'Personalized Palette', text: 'Get customized color recommendations based on the Monk Skin Tone (MST) scale.' },
              { icon: 'robot', title: 'AI-Powered Analysis', text: 'Leverage deep learning technology to accurately identify your skin tone category.' },
            ].map((feature) => (
              <div className="feature-card" key={feature.title}>
                <div className="feature-icon">
                  <i className={`fas fa-${feature.icon}`}></i>
                </div>
                <h3>{feature.title}</h3>
                <p>{feature.text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="mst-info">
        <div className="container">
          <div className="section-header">
            <h2>The Monk Skin Tone Scale</h2>
            <p>A more inclusive way to classify the rich diversity of human skin tones</p>
          </div>

          <div className="mst-showcase">
            <div className="mst-full-scale">
              {mstScaleItems.map((item) => (
                <div className="mst-item" style={{ backgroundColor: item.color }} key={item.mst}>
                  <span>{item.label}</span>
                </div>
              ))}
            </div>
            <div className="mst-description">
              <p>The Monk Skin Tone (MST) Scale is a ten-point skin tone scale designed to be more inclusive and representative of the diversity of human skin tones. Created by Harvard professor Dr. Ellis Monk, it helps ensure technology works well for people of all skin tones.</p>
              <ul className="mst-benefits">
                <li><i className="fas fa-check-circle"></i> More inclusive than traditional scales</li>
                <li><i className="fas fa-check-circle"></i> Scientifically developed and tested</li>
                <li><i className="fas fa-check-circle"></i> Used by leading technology companies</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section id="analyzer" className="analyzer">
        <div className="container">
          <div className="section-header">
            <h2>Analyze Your Skin Tone</h2>
            <p>Upload your photo to discover your skin tone classification and get personalized clothing recommendations</p>
          </div>

          <div className="analyzer-container">
            <div className="upload-card">
              <div className="card-header">
                <h3><i className="fas fa-cloud-upload-alt"></i> Upload Your Photo</h3>
                <p>For best results, use a well-lit photo of your face without makeup or filters</p>
              </div>

              <div
                className="upload-area"
                id="upload-area"
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(event) => {
                  event.preventDefault()
                  setDragActive(true)
                }}
                onDragLeave={(event) => {
                  event.preventDefault()
                  setDragActive(false)
                }}
                onDrop={handleDrop}
                style={dragActive ? { borderColor: 'var(--primary-color)' } : undefined}
              >
                <div className="upload-content">
                  <div className="upload-icon-container">
                    <i className="fas fa-cloud-upload-alt"></i>
                  </div>
                  <p>Drag and drop your image here</p>
                  <span>or</span>
                  <button className="browse-btn" type="button" onClick={(event) => {
                    event.stopPropagation()
                    fileInputRef.current?.click()
                  }}>
                    Browse Files
                  </button>
                  <p className="file-support">Supports: JPG, PNG, JPEG (Max 5MB)</p>
                </div>
                <input ref={fileInputRef} type="file" id="file-input" accept="image/*" hidden onChange={handleFileChange} />
              </div>

              {previewUrl && (
                <div className="preview-container" id="preview-container">
                  <div className="preview-header">
                    <h4>Image Preview</h4>
                    <button className="change-image-btn" id="change-image-btn" type="button" onClick={() => fileInputRef.current?.click()}>
                      <i className="fas fa-redo"></i> Change
                    </button>
                  </div>
                  <div className="image-preview" id="image-preview">
                    <img src={previewUrl} alt="Preview" />
                  </div>
                </div>
              )}

              <button id="analyze-btn" className="analyze-btn" disabled={!selectedFile || loading} onClick={handleAnalyze}>
                <i className="fas fa-magic"></i> Analyze Skin Tone
              </button>
              {errorMessage && (
                <div className="error-message">{errorMessage}</div>
              )}
            </div>

            {results && (
              <div className="results-card" id="results-section">
                <div className="card-header result-header">
                  <h3><i className="fas fa-chart-bar"></i> Analysis Results</h3>
                  <div className="confidence-badge" id="confidence">Confidence: {results.confidence.toFixed(2)}%</div>
                </div>

                <div className="result-content">
                  <div className="mst-result-container">
                    <h4>Your Skin Tone Classification</h4>
                    <div className="mst-scale-container">
                      <div className="mst-scale">
                        {mstScaleItems.map((item) => (
                          <div
                            className={`mst-color ${results.prediction === item.mst ? 'active' : ''}`}
                            data-mst={item.mst}
                            style={{ backgroundColor: item.color }}
                            key={item.mst}
                          >
                            <span>{item.label}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    <p id="mst-result" className="mst-result">Your skin tone is classified as {results.prediction} on the Monk Skin Tone Scale</p>
                    <div className="skin-type-group" id="skin-type-group">Skin Tone Group: {results.skinToneGroup.replace(/\b\w/g, (l) => l.toUpperCase())}</div>
                  </div>

                  <div className="face-detection-info" id="face-detection-info">
                    <div className={`detection-badge ${results.faceDetected ? 'success' : 'warning'}`}>
                      <i className={`fas fa-${results.faceDetected ? 'check-circle' : 'exclamation-triangle'}`}></i>
                      {results.faceDetected ? 'Face detected' : 'No face detection'}
                    </div>
                    <p>
                      {results.faceDetected
                        ? 'Analysis is performed on the detected face region and resized to the model input size.'
                        : 'Analysis is performed on the full image and resized to the model input size.'}
                    </p>
                    {results.processedImageUrl && (
                      <div className="processed-image-container">
                        <h4>Processed Image</h4>
                        <div className="processed-image">
                          <img src={results.processedImageUrl} alt="Processed" />
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="recommendations-container">
                    <h4>Your Personalized Color Recommendations</h4>

                    <div className="recommendation-tabs">
                      <button className={`tab-btn ${activeTab === 'recommended' ? 'active' : ''}`} onClick={() => setActiveTab('recommended')}>
                        Recommended Colors
                      </button>
                      <button className={`tab-btn ${activeTab === 'avoid' ? 'active' : ''}`} onClick={() => setActiveTab('avoid')}>
                        Colors to Avoid
                      </button>
                    </div>

                    <div className="tab-content">
                      <div className={`tab-pane ${activeTab === 'recommended' ? 'active' : ''}`} id="recommended-tab">
                        <div className="color-chips" id="recommended-colors">
                          {results.recommendations.recommended.map((color) => (
                            <div className="color-chip" key={color.name}>
                              <div className="color-preview" style={{ backgroundColor: color.hex }}></div>
                              <span>{color.name}</span>
                            </div>
                          ))}
                        </div>
                        <p className="recommendation-tip">These colors will enhance your natural skin tone and create a harmonious look.</p>
                      </div>

                      <div className={`tab-pane ${activeTab === 'avoid' ? 'active' : ''}`} id="avoid-tab">
                        <div className="color-chips" id="avoid-colors">
                          {results.recommendations.avoid.map((color) => (
                            <div className="color-chip" key={color.name}>
                              <div className="color-preview" style={{ backgroundColor: color.hex }}></div>
                              <span>{color.name}</span>
                            </div>
                          ))}
                        </div>
                        <p className="recommendation-tip">These colors may clash with your skin tone or make you appear washed out.</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      <section id="how-it-works" className="how-it-works">
        <div className="container">
          <div className="section-header">
            <h2>How It Works</h2>
            <p>Our advanced AI technology analyzes your photo in seconds</p>
          </div>

          <div className="steps-container">
            <div className="step">
              <div className="step-number">1</div>
              <div className="step-icon">
                <i className="fas fa-camera"></i>
              </div>
              <h3>Upload a Photo</h3>
              <p>Choose a well-lit, front-facing photo without makeup for the most accurate results.</p>
            </div>

            <div className="step-connector"></div>

            <div className="step">
              <div className="step-number">2</div>
              <div className="step-icon">
                <i className="fas fa-brain"></i>
              </div>
              <h3>AI Analysis</h3>
              <p>Our deep learning model analyzes your image and classifies your skin tone on the MST scale.</p>
            </div>

            <div className="step-connector"></div>

            <div className="step">
              <div className="step-number">3</div>
              <div className="step-icon">
                <i className="fas fa-palette"></i>
              </div>
              <h3>Get Recommendations</h3>
              <p>Receive personalized clothing color recommendations based on your skin tone classification.</p>
            </div>
          </div>
        </div>
      </section>

      <section id="faq" className="faq">
        <div className="container">
          <div className="section-header">
            <h2>Frequently Asked Questions</h2>
            <p>Find answers to common questions about our skin tone analyzer</p>
          </div>

          <div className="accordion">
            {[
              {
                question: 'How accurate is the skin tone analysis?',
                answer: 'Our AI model has been trained on diverse skin tone datasets and achieves high accuracy. However, results may vary based on lighting conditions, image quality, and other factors. For best results, use a well-lit photo without makeup or filters.',
              },
              {
                question: 'Is my photo stored or shared?',
                answer: 'Your privacy is important to us. Photos are processed locally in your browser and only uploaded to storage if configured. They are not shared with third parties.',
              },
              {
                question: 'How is the Monk Skin Tone scale different from other scales?',
                answer: 'The Monk Skin Tone (MST) scale was developed to be more inclusive of diverse skin tones. It features 10 shades that better represent the full spectrum of human skin colors, making it more comprehensive than traditional scales like Fitzpatrick which has only 6 categories.',
              },
              {
                question: 'Why does skin tone matter for clothing colors?',
                answer: 'Certain colors naturally complement different skin tones while others may clash or make you appear washed out. Understanding your skin tone helps you choose clothing colors that enhance your natural features and create a more harmonious appearance.',
              },
            ].map((item, index) => (
              <div className={`accordion-item ${accordionOpen[index] ? 'active' : ''}`} key={item.question}>
                <button className={`accordion-button ${accordionOpen[index] ? 'active' : ''}`} onClick={() => toggleAccordion(index)}>
                  <span>{item.question}</span>
                  <i className="fas fa-chevron-down"></i>
                </button>
                <div className="accordion-content">
                  <p>{item.answer}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="cta">
        <div className="container">
          <h2>Ready to Discover Your Perfect Colors?</h2>
          <p>Upload your photo now and transform your wardrobe with personalized color recommendations</p>
          <a href="#analyzer" className="cta-btn">Try Skin Tone Analyzer <i className="fas fa-arrow-right"></i></a>
        </div>
      </section>

      <footer>
        <div className="container">
          <div className="footer-content">
            <div className="footer-logo">
              <i className="fas fa-palette"></i> SkinTone<span>AI</span>
              <p>Discover your perfect color palette with AI-powered skin tone analysis</p>
            </div>

            <div className="footer-links">
              <h4>Quick Links</h4>
              <ul>
                <li><a href="#about">About</a></li>
                <li><a href="#analyzer">Analyzer</a></li>
                <li><a href="#how-it-works">How It Works</a></li>
                <li><a href="#faq">FAQ</a></li>
              </ul>
            </div>

            <div className="footer-links">
              <h4>Resources</h4>
              <ul>
                <li><a href="#">Privacy Policy</a></li>
                <li><a href="#">Terms of Service</a></li>
                <li><a href="#">Contact Us</a></li>
              </ul>
            </div>

            <div className="footer-newsletter">
              <h4>Stay Updated</h4>
              <p>Subscribe to our newsletter for style tips and updates</p>
              <div className="newsletter-form">
                <input type="email" placeholder="Your email address" />
                <button type="button"><i className="fas fa-paper-plane"></i></button>
              </div>
            </div>
          </div>

          <div className="footer-bottom">
            <p>&copy; 2025 SkinToneAI. All rights reserved.</p>
            <div className="social-links">
              <a href="#"><i className="fab fa-facebook-f"></i></a>
              <a href="#"><i className="fab fa-twitter"></i></a>
              <a href="#"><i className="fab fa-instagram"></i></a>
              <a href="#"><i className="fab fa-linkedin-in"></i></a>
            </div>
          </div>
        </div>
      </footer>

      <div className={`loading-overlay ${loading ? 'active' : ''}`} id="loading-overlay">
        <div className="loading-content">
          <div className="spinner"></div>
          <h3>Analyzing Your Image</h3>
          <p>This will only take a moment...</p>
        </div>
      </div>
    </div>
  )
}

export default App
