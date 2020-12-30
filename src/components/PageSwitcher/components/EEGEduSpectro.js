import React from "react";
import { catchError, multicast } from "rxjs/operators";

import { Card, RangeSlider, Button, ButtonGroup, TextContainer} from "@shopify/polaris";
import { Subject } from "rxjs";

import { zipSamples } from "muse-js";

import {
  bandpassFilter,
  epoch,
  fft
} from "@neurosity/pipes";

import * as generalTranslations from "./translations/en";

import Canvas from '../Canvas'
import * as funGAN from '../GAN'
import Webcam from "react-webcam"
import * as tf from '@tensorflow/tfjs';

// SETTINGS
let model_name = 'dcgan64'; //resnet128, dcgan64
let delay = 50; //msec between images in the morph sequence, can be low for 64, but should be 1000 for 128
let num_projections = 2; //number of latent projection of webcam image

export function getSettings () {
  return {
    cutOffLow: .01,
    cutOffHigh: 128,
    interval: 100,
    bins: 256,
    duration: 1024,
    srate: 256,
    name: 'EEG'
  }
};

//Setup model
let model_runner = new funGAN.ModelRunner();
model_runner.setup_model(model_name)

export function buildPipe(Settings) {
  if (window.subscriptionSpectro) window.subscriptionSpectro.unsubscribe();

  window.pipeSpectro$ = null;
  window.multicastSpectro$ = null;
  window.subscriptionSpectro = null;

  // Build Pipe
  window.pipeSpectro$ = zipSamples(window.source.eegReadings$).pipe(
    bandpassFilter({ 
      cutoffFrequencies: [Settings.cutOffLow, Settings.cutOffHigh], 
      nbChannels: window.nchans }),
    epoch({
      duration: Settings.duration,
      interval: Settings.interval,
      samplingRate: Settings.srate
    }),
    fft({ bins: Settings.bins }),
    catchError(err => {
      console.log(err);
    })
  );
  window.multicastSpectro$ = window.pipeSpectro$.pipe(
    multicast(() => new Subject())
  );
}

export function setup(setData, Settings) {

  console.log("Subscribing to " + Settings.name);

  if (window.multicastSpectro$) {
    window.subscriptionSpectro = window.multicastSpectro$.subscribe(data => {
      setData(spectroData => {
        Object.values(spectroData).forEach((channel, index) => {
          channel.datasets[0].data = data.psd[index];
          channel.xLabels = data.freqs
        });

        return {
          ch0: spectroData.ch0,
          ch1: spectroData.ch1,
          ch2: spectroData.ch2,
          ch3: spectroData.ch3,
          ch4: spectroData.ch4
        };
      });
    });

    window.multicastSpectro$.connect();
    console.log("Subscribed to " + Settings.name);
  }
}

function projectImage(inputImage, canvas) {
  console.log('Projecting image into GAN latent space on canvas: ' + canvas[0])
  return model_runner.project(model_name, inputImage, canvas)
}

export function renderModule(channels) {

  const videoConstraints = {
    width: { min: 256 },
    height: { min: 256 },
    aspectRatio: 1
  };

  const WebcamCapture = () => {
    const webcamRef = React.useRef(null);
    const [imgSrc, setImgSrc] = React.useState(null);
    const [tenSrc, setTenSrc] = React.useState(null);

    const capture = React.useCallback(() => {

        //get screenshot and set hook
        const imageSrc = webcamRef.current.getScreenshot();
        setImgSrc(imageSrc)

        //convert to image, then tensor, resize, and set hook
        var image = new Image();
        image.src = imageSrc;
        image.onload = function(){
          var tensorSrc = tf.browser.fromPixels(image);
          tensorSrc = tf.image.resizeBilinear(tensorSrc, [256, 256])
          setTenSrc(tensorSrc)
          console.log('Image aquired from webcam')

        }
      }, [webcamRef, setImgSrc, setTenSrc] // variables from inside scope coming out
    );

    //project the image from webcam into gan for each canvas
    const project = function() {
      for (let icanvas = 0; icanvas < num_projections; icanvas++) {
        projectImage(tenSrc, ["#" + icanvas])
      }     
    }
   
    return(
      <React.Fragment>
        <Card.Section>
            <TextContainer>
            <p> {[ "1) View webcam, line up face, and take photo" ]} </p>
            </TextContainer>
          {!imgSrc && (
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              videoConstraints={videoConstraints}
              width={256}
              height={256}
            />
          )}
          {imgSrc && (
            <img 
              src={imgSrc}
              alt={'dum'}
            />
          )}
          <ButtonGroup>
          <Button onClick={capture} disabled={imgSrc}
          >Capture photo</Button> 
          <Button onClick={project} disabled={!imgSrc}
          >Project Image</Button> 

          </ButtonGroup>
        </Card.Section>
      </React.Fragment>
    )
  }

  function RenderMorph() {
    Object.values(channels.data).map((channel, index) => {
      if (channel.datasets[0].data) { 

        //only left frontal channel
        if (index === 1) {
          window.psd = channel.datasets[0].data;
          window.freqs = channel.xLabels;
          if (channel.xLabels) {
            window.bins = channel.xLabels.length;
          } 
          if (window.freqs) {
            
            // timer to only animate psd sample every 'delay' 
            if (window.firstAnimate) {
              window.startTime = (new Date()).getTime();
              window.firstAnimate = false; 
            }
            let now = (new Date()).getTime();
            if (now - window.startTime > delay) {
              window.startTime =  (new Date()).getTime();

              //psd passed into the model generator function
              model_runner.generate(window.psd)
            }
          }
        }
      } 
    return null
    });
  }

  return (
    <React.Fragment>
      <Card >
        <Card.Section>
        {WebcamCapture()}
        {RenderMorph()}
        {[...Array(num_projections)].map((x, i) => 
          <Canvas canvas={["#" + i]} key={i} /> // loop to create multiple canvases
        )}    
        </Card.Section>
     
        <Card.Section>
          <TextContainer>
          <p> {[ "3) Then connect to EEG to morph face" ]} </p>
          </TextContainer>          
          <Canvas canvas="other_canvas"/>        
          <ButtonGroup>
            <Button
              primary = {window.psd}
              disabled={!window.psd}
              onClick={() => {
                model_runner.reseed(model_name)
              }}
            >
              {'Seed from Random Face'}
            </Button>
            <Button
              primary = {window.psd}
              disabled={!window.psd}
              onClick={() => {
                model_runner.webseed(model_name, num_projections)
              }}
            >
              {'Seed from Webcam Image'}
            </Button>            
          </ButtonGroup>
        </Card.Section>        
      </Card>
    </React.Fragment>
  );
}

export function renderSliders(setData, setSettings, status, Settings) {

  function resetPipeSetup(value) {
    buildPipe(Settings);
    setup(setData, Settings);
  }

  function handleIntervalRangeSliderChange(value) {
    setSettings(prevState => ({...prevState, interval: value}));
    resetPipeSetup();
  }

  function handleCutoffLowRangeSliderChange(value) {
    setSettings(prevState => ({...prevState, cutOffLow: value}));
    resetPipeSetup();
  }

  function handleCutoffHighRangeSliderChange(value) {
    setSettings(prevState => ({...prevState, cutOffHigh: value}));
    resetPipeSetup();
  }

  function handleDurationRangeSliderChange(value) {
    setSettings(prevState => ({...prevState, duration: value}));
    resetPipeSetup();
  }

  return (
    <Card title={Settings.name + ' Settings'} sectioned>
      <RangeSlider 
        disabled={status === generalTranslations.connect}
        min={128} step={128} max={4096} 
        label={'Epoch duration (Sampling Points): ' + Settings.duration} 
        value={Settings.duration} 
        onChange={handleDurationRangeSliderChange} 
      />
      <RangeSlider 
        disabled={status === generalTranslations.connect}
        min={10} step={5} max={Settings.duration} 
        label={'Sampling points between epochs onsets: ' + Settings.interval} 
        value={Settings.interval} 
        onChange={handleIntervalRangeSliderChange} 
      />
      <RangeSlider 
        disabled={status === generalTranslations.connect}
        min={.01} step={.5} max={Settings.cutOffHigh - .5} 
        label={'Cutoff Frequency Low: ' + Settings.cutOffLow + ' Hz'} 
        value={Settings.cutOffLow} 
        onChange={handleCutoffLowRangeSliderChange} 
      />
      <RangeSlider 
        disabled={status === generalTranslations.connect}
        min={Settings.cutOffLow + .5} step={.5} max={Settings.srate/2} 
        label={'Cutoff Frequency High: ' + Settings.cutOffHigh + ' Hz'} 
        value={Settings.cutOffHigh} 
        onChange={handleCutoffHighRangeSliderChange} 
      />
    </Card>
  )
}

