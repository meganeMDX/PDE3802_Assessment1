CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Inter+Tight:wght@700;800&display=swap');

.stApp{
  background:#d7e5f3;
  font-family:'Fira Code', ui-monospace, monospace;
  color:#0a0a0a;
}

header[data-testid="stHeader"]{
  background:transparent;
}

.block-container{
  max-width: 800px;
  padding-top:2rem;
  padding-bottom:4rem;
}

/* title and info button */
.app-title{
  font-family:'Inter Tight', sans-serif;
  font-weight:700;
  font-size:3rem;
  text-align:center;
}
.st-key-info_btn{
  display:flex;
  justify-content:flex-end;
  margin-top:22px;
}
.st-key-info_btn button{
  background:transparent!important;
  border:none;
  box-shadow:none;
  padding:.25rem;
  min-height:0;
}
.st-key-info_btn button p{
  display:none;
}
.st-key-info_btn button span[data-testid="stIconMaterial"]{
  font-size:2.5rem;
  color:#000;
}

/* panel for classes */
.classes-panel{
  background:linear-gradient(135deg,#eaf1fa 0%,#c3dcf3 32%,#b9d6f1 68%,#c9dff3 100%);
  border-radius:28px;
  padding:1.1rem 1.4rem 1.7rem;
  text-align:center;
  margin-top:.6rem;
}
.classes-panel h4{
  font-weight:400;
  font-size:1.06rem;
  margin:0 0 1.2rem;
}
.classes-grid{
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:0.65rem 10rem;
  width:fit-content;
  padding:0;
  text-align:left;
  margin: 0 auto;
}
.class-item{
  background:transparent;
  padding:0;
  margin:0;
  font-size:.95rem;
  letter-spacing:.03em;
  text-align:left;
  color: #444
}

/* image upload labels */
.upload-title{
  font-weight:700;
  font-size:1.5rem;
  text-align:center;
  margin:2.6rem 0 1rem;
}

/* upload image and live camera buttons */
[class*="st-key-mode_"] {
  width: 100%;
}
[class*="st-key-mode_"] button{
  min-height:3.3rem;
  border:none;
  border-radius:16px;
  box-shadow:none;
  color:#111;
  letter-spacing:.09em;
  font-size:1rem;
}
[class*="st-key-mode_"] button[data-testid="stBaseButton-secondary"]{
  background:#fff;
}
[class*="st-key-mode_"] button[data-testid="stBaseButton-primary"]{
  background:#8fb1d6
}

/* upload area */
.st-key-uploader [data-testid="stFileUploaderDropzone"]{
  background:#fff;
  border:1.5px dashed #111;
  border-radius:16px;
  min-height:290px;
  position:relative;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
}
.st-key-uploader [data-testid="stFileUploaderDropzoneInstructions"]{
  display:none;
}
.st-key-uploader [data-testid="stFileUploaderDropzone"]::before{
  content:"";
  width:118px;
  height:86px;
  background-repeat:no-repeat;
  background-position:center;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 1 24 22'%3E%3Cpath fill='%230070f0' d='M19.35 10.04A7.49 7.49 0 0 0 12 4C9.11 4 6.6 5.64 5.35 8.04A5.994 5.994 0 0 0 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96z'/%3E%3Cpath fill='white' d='M12 8l-4 4h3v4h2v-4h3z'/%3E%3C/svg%3E");
}
.st-key-uploader [data-testid="stFileUploaderDropzone"]::after{
  content:"Click or drag to upload image\A Supports JPG, JPEG and PNG";
  white-space:pre;
  text-align:center;
  font-size:1.15rem;
  line-height:1.9;
}
.st-key-uploader [data-testid="stFileUploaderDropzone"] button{
  position:absolute;
  inset:0;
  width:100%;
  height:100%;
  opacity:0;
  cursor:pointer;
}

/* analyse and clear buttons */
.st-key-analyse, .st-key-clear, .st-key-analyse button, .st-key-clear button{
  width:100%;
}
.st-key-analyse button, .st-key-clear button{
  min-height:3.3rem;
  border:none;
  border-radius:14px;
  width:100%;
}
.st-key-analyse button{
  background:#0070f0;
  color:#fff;
}
.st-key-clear button{
  background:#d9d9d9;
  color:#555;
}

/* box for image analysis */
.st-key-analysis_box{
  background:#fff;
  border:7px solid #000;
  border-radius:18px;
  padding:1rem;
  min-height:300px;
  display:flex;
  align-items:center;
  justify-content:center;
}
.analysis-image{
  display:block;
  max-width:100%;
  max-height:520px;
  margin:0 auto;
  border-radius:6px;
}
.image-reveal{
  animation:revealAnimation .6s ease-out;
}
/* animation for reveal of image */
@keyframes revealAnimation{
  from{opacity:0; filter:blur(8px);}
  to{opacity:1; filter:none;}
}

/* animation for scanning */
.image-scan{
  position:relative;
  display:inline-block;
  overflow:hidden;
  border-radius:8px;
}
.image-scan img{
  display:block;
  max-width:100%;
  max-height:480px;
}
.image-scan-line{
  position:absolute;
  left:0;
  right:0;
  top:0;
  height:4px;
  background:#0070f0;
  box-shadow:0 0 20px 5px rgba(0,112,240,0.5);
  animation:scanAnimation 1.5s ease-in-out infinite alternate;
}
@keyframes scanAnimation{
  from{top:0;}
  to{top:calc(100% - 4px);}
}
.image-scan-wrap{
  text-align:center;
  width:100%;
}
.process-stages{
  display:flex;
  justify-content:center;
  gap:.5rem;
  margin-top:1rem;
}
.process-stage{
  padding:.3rem .8rem;
  border-radius:99px;
  background:#e8eef5;
  color:#777;
  font-size:.85rem;
}
.process-stage.now{
  background:#0070f0;
  color:#fff;
}

/* result section */
.classification-result{
  background:#9fc1de;
  border-radius:28px;
  padding:1.3rem 2.2rem 1.7rem;
  margin:2.2rem auto 0;
  max-width:640px;
}
.classification-result-title{
  text-align:center;
  font-weight:700;
  text-decoration:underline;
  font-size:1.2rem;
}
.class-result-label{
  text-align:left;
  color:#0070f0;
  font-weight:700;
  font-size:1.4rem;
  margin:0;
}
.confidence-row{
  display:flex;
  justify-content:space-between;
  margin-bottom:.35rem;
  font-size:0.9rem;
}
.confidence-bar{
  background:#fff;
  border-radius:99px;
  height:20px;
  overflow:hidden;
}
.confidence-bar-fill{
  background:#0070f0;
  height:100%;
  animation:growAnimation 1.1s ease-out;
}

@keyframes growAnimation{
  from{width:0!important;}
}

.confidence-level{
  display:inline-block;
  padding:.2rem .8rem;
  border-radius:99px;
  background:#fff;
  font-size:.85rem;
  margin-top:.9rem;
}

.confidence-level.high{
  background:#DDF3E4;
  color: #217346
}

.confidence-level.moderate{
  background:#FFF0CC;
  color: #805B00
}

.confidence-level.low{
  background:#FCE2E2;
  color: #A32626
}

.class-result-item{
  padding: 5px 0;
}

.class-result-item:not(:first-child){
  border-top: 1px solid #d9e2ec;
  margin-top:12px;
}

/* popup for model info */
div[data-testid="stDialog"] div[role="dialog"]{
  background:#9ebcff;
  border-radius:34px;
}
div[data-testid="stDialog"] h2{
  text-align:center;
  font-weight:700;
  width:100%;
  margin:0;
  padding-bottom:0;
}
.sub-heading{
  font-weight:700;
  font-size:1.25rem;
  margin:.9rem 0 .4rem;
}
.model-info-body{
  line-height:1.7;
  font-size:.98rem;
}
.metrics-grid{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
  gap:.9rem;
  margin:1.4rem 0;
}
.metric-card{
  background:#fff;
  border-radius:16px;
  padding:.8rem 1rem;
  text-align:center;
}
.metric-card b{
  display:block;
  font-size:1.35rem;
  color:#0070f0;
}
.chart-placeholder{
  border:1.5px dashed #345;
  border-radius:14px;
  min-height:170px;
  display:flex;
  align-items:center;
  justify-content:center;
  text-align:center;
  padding:1rem;
  background:rgba(255,255,255,.35);
}

button:disabled, button:disabled:hover{
  background:#c5d3e1 !important;
  color:#65758a !important;
  opacity:1 !important;
}
"""