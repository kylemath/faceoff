import React from "react";
import { catchError, multicast } from "rxjs/operators";

import { Card, RangeSlider} from "@shopify/polaris";
import { Subject } from "rxjs";

import { zipSamples } from "muse-js";

import {
  bandpassFilter,
  epoch,
  fft
} from "@neurosity/pipes";

import { chartStyles } from "./chartOptions";

import * as generalTranslations from "./translations/en";

import Canvas from '../Canvas'

import * as funGAN from '../GAN'

let model_runner = new funGAN.ModelRunner();
let model_name = 'dcgan64';

export function getSettings () {
  return {
    cutOffLow: 2,
    cutOffHigh: 50,
    interval: 16,
    bins: 256,
    duration: 256,
    srate: 256,
    name: 'Spectro',
    sliceFFTLow: 1,
    sliceFFTHigh: 100,
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
    // sliceFFT([Settings.sliceFFTLow, Settings.sliceFFTHigh]),
    catchError(err => {
      console.log(err);
    })
  );
  window.multicastSpectro$ = window.pipeSpectro$.pipe(
    multicast(() => new Subject())
  );
}

export function setup(setData, Settings) {

  model_runner.setup_model(model_name)
  model_runner.generate();

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

export function renderModule(channels) {

  function RenderCharts() {
    return Object.values(channels.data).map((channel, index) => {
      if (channel.datasets[0].data) {
        window.psd = channel.datasets[0].data;
        window.freqs = channel.xLabels;
        if (channel.xLabels) {
          window.bins = channel.xLabels.length;
        }
      }   

      //only left frontal channel
      if (index === 1 && window.freqs) {
        model_runner.generate(window.psd);

        return (
          <React.Fragment key={'dum'}>
            <Card.Section>
              <Canvas />
            </Card.Section>
          </React.Fragment>
        );
      } else {
        return null
      }
    });
  }

  return (
    <Card >
      <Card.Section>
        <div style={chartStyles.wrapperStyle.style}>{RenderCharts()}</div>
      </Card.Section>
    </Card>
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

