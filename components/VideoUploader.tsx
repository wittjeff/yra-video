"use client";

import React, { useState } from "react";
import axios from "axios";

const VideoUploader: React.FC = () => {

  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [captionUrl, setCaptionUrl] = useState<string | null>(null);
  const [vttContent, setVttContent] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append("video", file);

    try {
      const uploadResponse = await axios.post("/api/upload", formData);
      setUploading(false);
      setProcessing(true);

      const processResponse = await axios.post("/api/process", {
        filename: uploadResponse.data.filename,
      });
      setProcessing(false);
      setCaptionUrl(processResponse.data.captionUrl);

      setVttContent(processResponse.data.vttContent);
      if (processResponse.data.captionUrl) {
        setCaptionUrl(processResponse.data.captionUrl);
      } 
    } catch (error) {
      console.error("Error:", error);
      setUploading(false);
      setProcessing(false);
    }
  };

  return (
    <div className="container">
      <h1 className="title">Transcribe your videos.</h1>
      <p className="subtitle">
        Just upload your video or audio and Whisper API will do the rest. Also,
        you can translate your transcription to listed languages.
      </p>
      <div className="upload-section">
        <div className="file-input-wrapper">
          <input
            type="file"
            accept="video/*,audio/*"
            onChange={handleFileChange}
            id="file-input"
          />
          <label htmlFor="file-input" className="file-input-label">
            Choose file
          </label>
          <span className="file-name">
            {file ? file.name : "No file chosen"}
          </span>
        </div>
        <button
          onClick={handleUpload}
          disabled={!file || uploading || processing}
          className="upload-button"
        >
          Upload and Process
        </button>
      </div>
      {uploading && <p className="status">Uploading...</p>}
      {processing && (
        <p className="status">Processing video and generating captions...</p>
      )}
      {captionUrl && (
        <a
          href={captionUrl}
          download
          className="download-link"
          onClick={(e) => {
            e.preventDefault(); // Prevent default link behavior
            console.log("Download link clicked");
            console.log("Caption URL on click:", captionUrl);
            // Force a server request
            window.location.href = captionUrl;
          }}
        >
          Download Captions
        </a>
      )}
      {vttContent && (
        <div className="vtt-content">
          <h3>VTT Content:</h3>
          <pre>{vttContent}</pre>
        </div>
      )}
      <style jsx>{`
        .container {
          max-width: 800px;
          margin: 0 auto;
          padding: 20px;
          font-family: Arial, sans-serif;
        }
        .title {
          font-size: 2.5em;
          margin-bottom: 10px;
        }
        .subtitle {
          color: #666;
          margin-bottom: 20px;
        }
        .upload-section {
          display: flex;
          flex-direction: column;
          gap: 10px;
          margin-bottom: 20px;
        }
        .file-input-wrapper {
          display: flex;
          align-items: center;
          background-color: #f1f3f4;
          border-radius: 4px;
          overflow: hidden;
        }
        .file-input-wrapper input[type="file"] {
          display: none;
        }
        .file-input-label {
          background-color: #e8eaed;
          color: #3c4043;
          padding: 10px 15px;
          cursor: pointer;
          font-weight: bold;
        }
        .file-name {
          padding: 10px 15px;
          color: #3c4043;
        }
        .upload-button {
          padding: 10px 20px;
          background-color: #007bff;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 1em;
        }
        .upload-button:disabled {
          background-color: #cccccc;
        }
        .status {
          text-align: center;
          color: #666;
        }
        .download-section {
          text-align: center;
          margin-bottom: 20px;
        }
        .download-link {
          display: inline-block;
          padding: 10px 20px;
          background-color: #28a745;
          color: white;
          text-decoration: none;
          border-radius: 5px;
        }
        .vtt-content {
          background-color: #f8f9fa;
          border: 1px solid #dee2e6;
          border-radius: 5px;
          padding: 20px;
          margin-top: 20px;
        }
        .vtt-content h3 {
          margin-top: 0;
          color: #333;
        }
        .vtt-content pre {
          white-space: pre-wrap;
          word-wrap: break-word;
          max-height: 400px;
          overflow-y: auto;
        }
      `}</style>
    </div>
  );
};

export default VideoUploader;
