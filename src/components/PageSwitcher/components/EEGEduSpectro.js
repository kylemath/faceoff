import React, { useState, useCallback } from "react";
import { catchError, multicast } from "rxjs/operators";

import { Card, RangeSlider, Button, ButtonGroup, TextContainer, Select} from "@shopify/polaris";
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

import { FullScreen, useFullScreenHandle } from "react-full-screen"

let model_runner = new funGAN.ModelRunner();

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

export function getLearningSettings () {
  return {
    learningRate: .05,
    trainingSteps: 150, 
    stepsPerImage: 5,
    numProjections: 1, //number of latent projection of webcam image
    dampingOfChange: 10, //smaller is more change
    morphDelay: 50, //msec between images in the morph sequence, can be low for 64, but should be 1000 for 128
    modelName: 'dcgan64'
  }
};


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

const chartTypes = [
  { label: 'dcgan64', value: 'dcgan64'},
  { label: 'resnet128', value: 'resnet128'}, 
  { label: 'resnet256', value: 'resnet256'}
];

export function RenderModule(channels) {

  const videoConstraints = {
    width: { min: 256 },
    height: { min: 256 },
    aspectRatio: 1
  };

  const [learningSettings, setLearningSettings] = React.useState(getLearningSettings)

  // for picking a new module
  const [selected, setSelected] = useState('dcgan64');
  const handleSelectChange = useCallback(value => {
    setSelected(value);
    learningSettings.modelName = selected;

  }, [learningSettings, selected]);

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

    //setup model
    const setupModel = function() {
      model_runner.setup_model(learningSettings.modelName)
    }     

    const projectImage = function(inputImage, canvas, settings) {
      console.log('Projecting image into GAN latent space on canvas: ' + canvas[0])
      console.log(learningSettings.modelName)
      return model_runner.project(learningSettings.modelName, inputImage, canvas, settings)
    }

    //project the image from webcam into gan for each canvas
    const project = function() {
      for (let icanvas = 0; icanvas < learningSettings.numProjections; icanvas++) {
        projectImage(tenSrc, ["#" + icanvas], learningSettings)
      }     
    }

    function handleLearningRateRangeSliderChange(value) {
      setLearningSettings(prevState => ({...prevState, learningRate: value}));
    }

    function handleTrainingStepsRangeSliderChange(value) {
      setLearningSettings(prevState => ({...prevState, trainingSteps: value}));
    }   

    function handleStepsPerImageRangeSliderChange(value) {
      setLearningSettings(prevState => ({...prevState, stepsPerImage: value}));
    }   

    function handleNumProjectionsRangeSliderChange(value) {
      setLearningSettings(prevState => ({...prevState, numProjections: value}));
    }   

    return(
      <React.Fragment>
        <Card.Section>
            <Select
              label={""}
              options={chartTypes}
              onChange={handleSelectChange}
              value={selected}
            />
            <Button onClick={setupModel}>Setup model</Button>
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

          </ButtonGroup>
        </Card.Section>
        <Card.Section>
          <TextContainer>
            2) Then project your picture into the latent space
          </TextContainer>
          <RangeSlider 
            disabled={window.isprojecting}
            min={10} step={10} max={500} 
            label={'Training Steps: ' + learningSettings.trainingSteps} 
            value={learningSettings.trainingSteps} 
            onChange={handleTrainingStepsRangeSliderChange} 
          />          
          <RangeSlider 
            disabled={window.isprojecting}
            min={.001} step={.001} max={.1} 
            label={'Optimizer Learning Rate: ' + learningSettings.learningRate} 
            value={learningSettings.learningRate} 
            onChange={handleLearningRateRangeSliderChange} 
          />
          <RangeSlider 
            disabled={window.isprojecting}
            min={2} step={1} max={learningSettings.trainingSteps} 
            label={'Plotting frequency (every n images): ' + learningSettings.stepsPerImage} 
            value={learningSettings.stepsPerImage} 
            onChange={handleStepsPerImageRangeSliderChange} 
          />
          <RangeSlider 
            disabled={window.isprojecting}
            min={1} step={1} max={10} 
            label={'Number of parallel projections: ' + learningSettings.numProjections} 
            value={learningSettings.numProjections} 
            onChange={handleNumProjectionsRangeSliderChange} 
          />                        
          <Button onClick={project} disabled={!imgSrc}
          >Project Image</Button> 
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
            if (now - window.startTime > learningSettings.morphDelay) {
              window.startTime =  (new Date()).getTime();

              if (window.thisFace) {
                //psd passed into the model generator function
                model_runner.generate(window.psd, learningSettings)
              }
            }
          }
        }
      }
      return null
    });

    function handleDampingOfChangeRangeSliderChange(value) {
      setLearningSettings(prevState => ({...prevState, dampingOfChange: value}));
    }

    function handleMorphDelayRangeSliderChange(value) {
      setLearningSettings(prevState => ({...prevState, morphDelay: value}));
    }

    return( 
      <React.Fragment>      
        <RangeSlider 
          disabled={window.isprojecting}
          min={1} step={1} max={100} 
          label={'Damping Of Morphing Change: ' + learningSettings.dampingOfChange} 
          value={learningSettings.dampingOfChange} 
          onChange={handleDampingOfChangeRangeSliderChange} 
        />
        <RangeSlider 
          disabled={window.isprojecting}
          min={50} step={50} max={1000} 
          label={'Morphing Frequency (ms): ' + learningSettings.morphDelay} 
          value={learningSettings.morphDelay} 
          onChange={handleMorphDelayRangeSliderChange} 
        />        
      </React.Fragment>
    )
  }

  const handle = useFullScreenHandle();

  return (
    <React.Fragment>
      <Card >
        <Card.Section>
        {WebcamCapture()}
        {[...Array(learningSettings.numProjections)].map((x, i) => 
          <Canvas canvas={["#" + i]} key={i} /> // loop to create multiple canvases
        )} 
   
        </Card.Section>
     
        <Card.Section>
          <TextContainer>
          <p> {[ "3) Then connect to EEG to morph face" ]} </p>
          </TextContainer>          
          {RenderMorph()}

          <FullScreen handle={handle} >
            <Canvas canvas="other_canvas"/>  
          </FullScreen>      

          <ButtonGroup>
            <Button
              primary = {window.psd}
              disabled={!window.psd}
              onClick={() => {
                model_runner.reseed(learningSettings.modelName)
              }}
            >
              {'Seed from Random Face'}
            </Button>
            <Button
              primary = {window.psd}
              disabled={!window.psd | !window.tfout["#0"]}
              onClick={() => {
                model_runner.webseed(learningSettings.modelName, learningSettings.numProjections)
              }}
            >
              {'Seed from Webcam Best Fit'}
            </Button>  
          <Button onClick={handle.enter}
                  disabled={!window.psd}
          >
           Enter Fullscreen
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

