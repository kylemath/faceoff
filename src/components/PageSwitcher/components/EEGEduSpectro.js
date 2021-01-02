import React, { useState, useCallback, useEffect } from "react";
import { catchError, multicast } from "rxjs/operators";

import { Card, RangeSlider, Button, ButtonGroup, TextContainer, Select, Link} from "@shopify/polaris";
import { Subject } from "rxjs";

import { zipSamples } from "muse-js";

import {
  bandpassFilter,
  epoch,
  fft
} from "@neurosity/pipes";

import * as generalTranslations from "./translations/en";

import Canvas from '../Canvas'
import CanvasFinal from '../CanvasFinal'
import * as funGAN from '../GAN'
import Webcam from "react-webcam"
import * as tf from '@tensorflow/tfjs';

import { FullScreen, useFullScreenHandle } from "react-full-screen"

let model_runner = new funGAN.ModelRunner();
model_runner.setup_model('dcgan64')

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
    dampingOfChange: 20, //smaller is more change
    morphDelay: 25, //msec between images in the morph sequence, can be low for 64, but should be 1000 for 128
    modelName: 'dcgan64'
  }
};

window.step = 0; 

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


    const strings = {
      "background1": [
        "Neural Networks can be trained to understand faces, using millions of interconnected computational units. ",
        "We use these tools every day in our smartphones and social media where they can identify and recognize faces. ",
        "Neuroscientists are interested in these artificial brain networks as a useful model for how our own brains represent visual information like faces. ",
        "By studying how these artifical brains represent things we are familiar with like faces, we can better understand how our own brains might do it."
      ],
      "background2": [
        "One facinating type of neural network is called a Generative Adversarial Network, or GAN. ",
        "These networks are essential a pair of competing artificial neural networks, one trained to JUDGE if an image is from a training set (a face for example), and one trained to GENERATE examples to try to trick the JUDGE. ", 
        "Using the modern tools of deep machine learning, these two large neural networks can be trained extensively in tandem until the generator can create COMPLETELY NOVEL examples from the set of stimuli in the training set."
      ], 
      "background3": [
        "Three of these models are included here. Each can generate novel faces when given an array of 128 random numbers. However, a warning is in order. ",
        "THIS PROJECT IS FROM THE FUTURE. The computational task is too large for most computers at anything better than lower settings. ",
        "Therefore, although larger models are included if you are patient and willing to play around with settings, until our computers improve we suggest you stick with the 64x64 dcgan model. ",
        "Start by setting up the model you want here:"
       ]      

    }

    return(
      <React.Fragment>
        <Card.Section title={'Brains that make their own faces'}>
          <TextContainer>
            <p> {strings.background1} </p>
            <p> {strings.background2} </p>
            <p></p>            
            <Link url="https://en.wikipedia.org/wiki/Generative_adversarial_network" target="_blank"> Learn more about GANs</Link>
            <p> {strings.background3} </p>
            <p></p>            
          </TextContainer>
          <Select
            label={""}
            options={chartTypes}
            onChange={handleSelectChange}
            value={selected}
          />
          <Button onClick={setupModel}>Setup model</Button>
          <p style={{color:"white"}}>{ "|"}</p>            
        </Card.Section>
        <Card.Section title={'Connect webcam and take photo'}>
          <TextContainer>
            <p> {
              [ 
              "Next you will allow your webcam access, line up face to fill the frame, and capture a photo. ",
              "Make sure you have good lighting and not much in the background." ]
            } </p>
            <p style={{color:"white"}}>{ "|"}</p>            
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
        <Card.Section title={'Project your face into the GAN model'}>
          <TextContainer>
            <p>{[
              "Next we will digitize this face in the GAN model you selected. The GENERATOR of the GAN takes an input of 128 numbers as a barcode for each face it creates. ",
              "We want to adjust these 128 numbers to make them create a face that looks alot like the picture you just took. ", 
              "We call this projecting your picture into the latent space since since we are trying to find a vector of values in the model, a barcode, that best matches your face. ",
              "We do this again using a modern deep learning technique called gradient descent. We generate a random image to start from, convert it to a picture of a face, and then find the difference of every pixel between this generated image and your webcam image. ",
              "This sum of differences in pixels is called our Loss Function. This is what we want to get smaller as we change the image to more closely look like the webcam image. ",
              "We use this loss function to find the optimal way to modify the 128 numbers to minimize this difference, such that the face starts to look to you." 
              ]
            }
            </p>
            <p>
            {[
              "These default settings should work, but you can adjust how many parallel faces it tries to make look like you at the bottom. ",
              "Each face will add a great deal more computational cost so add faces sparingly. They will get averaged together in the next step. ",
              "Click Project Image when you are ready to start the projection and watch the progress images and slider. "
              ]}
            </p>
           <p></p>            
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
            min={2} step={10} max={learningSettings.trainingSteps} 
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
          <p style={{color:"white"}}>{ "|"}</p>                
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
          label={'Damping Of Morphing Change (<-- more change): ' + learningSettings.dampingOfChange} 
          value={learningSettings.dampingOfChange} 
          onChange={handleDampingOfChangeRangeSliderChange} 
        />
        <RangeSlider 
          disabled={window.isprojecting}
          min={10} step={10} max={1000} 
          label={'Morphing Frequency (ms): ' + learningSettings.morphDelay} 
          value={learningSettings.morphDelay} 
          onChange={handleMorphDelayRangeSliderChange} 
        />        
      </React.Fragment>
    )
  }

  const handle = useFullScreenHandle();
  const keyFunction = useCallback((event) => {
    if(event.keyCode === 82) {  //r
      console.log('You pressed R')
      model_runner.reseed(learningSettings.modelName)

    }
    if(event.keyCode === 87) {  //w
      console.log('You pressed W')
      model_runner.webseed(learningSettings.modelName, learningSettings.numProjections)

    }
  }, [learningSettings]);

  useEffect(() => {
    document.addEventListener("keydown", keyFunction, false);

    return () => {
      document.removeEventListener("keydown", keyFunction, true);
    };
  }, [keyFunction]);

  const styles = {
    color:"white"
  }

  function handleTrainingStepRangeSliderChange(value) {
      console.log('TrainingStepSliderHandle', value)
  }

  return (
    <React.Fragment>
      <Card >
        <Card.Section>
        {WebcamCapture()}
        {[...Array(learningSettings.numProjections)].map((x, i) => 
          <Canvas canvas={["#" + i]} key={i} /> // loop to create multiple canvases
        )} 
        <RangeSlider 
          disabled={!window.thisFace}
          min={1} step={1} max={learningSettings.trainingSteps} 
          label={'Training Step: ' + window.step} 
          value={window.step} 
          onChange={handleTrainingStepRangeSliderChange} 
        />

        </Card.Section>
     
        <Card.Section title={'Morph the face with your brain'}>
          <TextContainer>
          <p> {[
                "Finally we can use the EEG to morph this digital version of your face, or a random face. ",
                "Change the first slider to make the face morph more or less. Change the second slider to adjust how often the face changes. ",
                "For the 64x64 model this can be as low as 10ms, but for 128x it should be at least 1000 ms depending on hardware. ",
                "Once your EEG is connected above, and your face is projected, the blue bottons should be available, and you can enter full screen."
              ]} </p>
          </TextContainer>          
          <p style={{color:"white"}}>{ "|"}</p>            
          
          {RenderMorph()}

          <FullScreen handle={handle} >
            <CanvasFinal canvas="other_canvas"/>  
            <p style={styles}>Press R to reseed from random, W to reseed from webcam </p>
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
    <Card title={'EEG Settings'} sectioned>
      <TextContainer>
      <p> {[
        'You may ask, what about the EEG is morphing the image? We are taking one of the channels, on the left forehead, and using only its data. ',
        "We are using a Fast Fourier Transform to convert segments of this data into into the frequency domain, to represent the voltage over time as power at 128 different frequencies between 1 and 128 Hz. ",
        "Therefore, one interesting thing you can try is using narrow band filters by adjusting the cutoffs below. ",
        "How do different frequency bands influence the image? Each of the 128 values in the barcode represents different aspects of the face after all."

      ]} </p>
     <p></p>            

     <Link url="https://eegedu.com"> Learn more about EEG with the Muse at EEGedu.com</Link>

      </TextContainer>
      <p style={{color:"white"}}>{ "|"}</p>            

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

