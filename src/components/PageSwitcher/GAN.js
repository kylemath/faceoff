import * as tf from '@tensorflow/tfjs';

// Gan Stuff

let all_model_info = {
    dcgan64: {
        description: 'DCGAN, 64x64 (16 MB)',
        model_url: "https://storage.googleapis.com/store.alantian.net/tfjs_gan/chainer-dcgan-celebahq-64/tfjs_SmoothedGenerator_50000/model.json",
        model_size: 64,
        model_latent_dim: 128,
        draw_multiplier: 4,
        animate_frame: 200,
    },
    resnet128: {
        description: 'ResNet, 128x128 (252 MB)',
        model_url: "https://storage.googleapis.com/store.alantian.net/tfjs_gan/chainer-resent128-celebahq-128/tfjs_SmoothedGenerator_20000/model.json",
        model_size: 128,
        model_latent_dim: 128,
        draw_multiplier: 2,
        animate_frame: 10
    },
    resnet256: {
        description: 'ResNet, 256x256 (252 MB)',
        model_url: "https://storage.googleapis.com/store.alantian.net/tfjs_gan/chainer-resent256-celebahq-256/tfjs_SmoothedGenerator_40000/model.json",
        model_size: 256,
        model_latent_dim: 128,
        draw_multiplier: 1,
        animate_frame: 10
    }
};

function image_enlarge(y, draw_multiplier) {
    if (draw_multiplier === 1) {
        return y;
    }
    let size = y.shape[0];
    return y.expandDims(2).tile([1, 1, draw_multiplier, 1]
    ).reshape([size, size * draw_multiplier, 3]
    ).expandDims(1).tile([1, draw_multiplier, 1, 1]
    ).reshape([size * draw_multiplier, size * draw_multiplier, 3])
}

let dampingOfChange = 10; //smaller is more change

function getRandomInt(max) {
  return Math.floor(Math.random() * Math.floor(max));
}

// Delay that many ms before tf computing, which can block UI drawing.
const ui_delay_before_tf_computing_ms = 20; 

function resolve_after_ms(x, ms) {
    return new Promise(resolve => {
        setTimeout(() => {
            resolve(x);
        }, ms);
    });
}

async function computing_generate_main(model, size, draw_multiplier, latent_dim, psd) {
    if (psd) {
        const zNormalized = tf.tidy(() => {
            //convert psd to tensor
            const z = tf.tensor(psd, [1, latent_dim])

            //compute mean and variance
            const psdMean = z.mean();
            const diffFromMean = z.sub(psdMean);
            const squaredDiffFromMean = diffFromMean.square();
            const variance = squaredDiffFromMean.mean();
            const psdSD = variance.sqrt();

            //subtract mean and divide by SD to normalize
            var zMeanSubtract = z.sub(psdMean);
            var zNormalized = zMeanSubtract.div(psdSD).div(dampingOfChange);
            return zNormalized;
        });

        // to avoid moving off into the edge of hyperspace, sometimes add, sometimes subtract
        if (getRandomInt(2) === 0) {
            window.thisFace = window.thisFace.add(zNormalized);
        } else {
            window.thisFace = window.thisFace.sub(zNormalized);
        }

        const y = model.predict(window.thisFace).squeeze().transpose([1, 2, 0]).div(tf.scalar(2)).add(tf.scalar(0.5));
        const outPixels = image_enlarge(y, draw_multiplier);
        // let c = document.getElementById("the_canvas");
        // await tf.browser.toPixels(outPixels, c);
        let d = document.getElementById("other_canvas");
        await tf.browser.toPixels(outPixels, d);        
    };
}


const tensor_length = function(tensor, dim) {
    return tf.abs(tf.sub(tf.norm(tensor), tf.sqrt(dim)))
}


window.tfout = {};
async function computing_fit_target_latent_space(model, draw_multiplier, latent_dim, input_image, canvas) {
    console.log('Finding the closest vector in latent space on canvas: ', canvas[0]);

    // Define the two canvas names
    let the_canvas = document.getElementById(canvas);
    // let the_other_canvas = document.getElementById("the_other_canvas");

    // Get the generated image from other canvas and convert to tensor
    // target_image is a Uint8ClampedArray
    // target_image_tensor is a [256, 256, 3] Int32Array, range [0, 255]
    // var ctx = the_other_canvas.getContext("2d");
    // let target_image = ctx.getImageData(0, 0, 256, 256);


    let target_image_tensor = input_image;


    // Create new random vector in latent space to start from
    // z is sampled from normal distribution
    // mean of 0, standard deviation of 1
    const z = tf.variable(tf.randomNormal([1, latent_dim]));
  
    const generate_and_enlarge_image = function(first_time) {
        // rescale only on first generation by multiplying by 255
        const scaler = (first_time === true) ? 255 : 1

        // model outputs values between [-1, 1]
        const y_unnormalize = model.predict(z).squeeze().transpose([1, 2, 0]);
        // scale the values to the range [0, 1]
        const y_small = y_unnormalize.div(tf.scalar(2)).add(tf.scalar(0.5)).mul(scaler);
        // Enlarge the image to fit the canvas
        let y = image_enlarge(y_small, draw_multiplier);

        return y;
    }

    const loss = function(pred, label) {
        return pred.sub(label).abs().mean();
    }

    const _loss_function = function() {
        // Generate Random image and compute loss
        const first_time = true;
        const predicted_image_tensor = generate_and_enlarge_image(first_time);

        var computed_loss = loss(predicted_image_tensor, target_image_tensor);

        // computed_loss.data().then(l => {
        //     console.log('Loss: ', l[0]);
        // });

        // Add regularization to the loss function
        const regularize = tensor_length(z, latent_dim);
        computed_loss = tf.add(computed_loss, regularize);

        return computed_loss
    }

    // Define an optimizer
    const learningRate = 0.05;
    const optimizer = tf.train.adam(learningRate);

    const num_steps = 20;
    const steps_per_image = 2;

    // Train the model.
    for (let i = 0; i < num_steps; i++ ) {

        // Compute and apply gradients
        let {value, grads} = optimizer.computeGradients(_loss_function, [z]);
        value.data().then(l => {
            console.log('Canvas: ', canvas[0], ', Training Step: ', i, 'Loss: ', Number.parseFloat(l[0]).toPrecision(5))
        })
        optimizer.applyGradients(grads);

        // put image on canvas periodically and at end
        if (i % steps_per_image === 0 | i === num_steps) {

            // Generate the new best image
            let first_time = false;
            let y = generate_and_enlarge_image(first_time) 

            // Print it to the top canvas
            await tf.browser.toPixels(y, the_canvas);
        }
    }

    //save to window to load into morpher

    window.tfout[canvas] = z;
}

export class ModelRunner {
    constructor() {
        this.model_promise_cache = {};
        this.model_promise = null;
        this.model_name = null;
        this.start_time = null;
    }

    setup_model(model_name) {
        this.model_name = model_name;
        let model_info = all_model_info[model_name];
        let model_url = model_info.model_url,
            model_latent_dim = model_info.model_latent_dim,
            description = model_info.description;


        console.log(`Setting up model ${description}`);

        if (model_name in this.model_promise_cache) {
            this.model_promise = this.model_promise_cache[model_name];
        } else {
            this.model_promise = tf.loadLayersModel(model_url);
            window.thisFace = tf.randomNormal([1, model_latent_dim]);

            this.model_promise.then((model) => {
                return resolve_after_ms(model, ui_delay_before_tf_computing_ms);
            }).then((model) => {
                console.log(`Model "${description} is ready.`);
            });
            this.model_promise_cache[model_name] = this.model_promise;
        }
    }

    reseed(model_name) {
        this.model_name = model_name;
        let model_info = all_model_info[model_name];
        let model_latent_dim = model_info.model_latent_dim;

        console.log(`Reseeding model `);
        window.thisFace = tf.randomNormal([1, model_latent_dim]);
    }

    webseed(model_name, num_projections) {
        this.model_name = model_name;

        console.log(`Seeding model from Webcam image `);
        // Replace with something like
        
        for (var key in window.tfout) {
            if (key === "the_canvas0") {
                window.thisFace = window.tfout[key]
            } else {          
                window.thisFace = tf.add(window.thisFace, window.tfout[key])
            }
        }
        window.thisFace = tf.div(window.thisFace, tf.scalar(num_projections) )

    }

    project(model_name, input_image, canvas) {
        let model_info = all_model_info[this.model_name];
        let model_latent_dim = model_info.model_latent_dim,
            draw_multiplier = model_info.draw_multiplier;
       
        this.model_promise.then((model) => {
            return resolve_after_ms(model, ui_delay_before_tf_computing_ms);
        }).then((model) => {
            return computing_fit_target_latent_space(model, draw_multiplier, model_latent_dim, input_image, canvas)        
        });
        

    }

    generate(psd) {
        let model_info = all_model_info[this.model_name];
        let model_size = model_info.model_size,
            model_latent_dim = model_info.model_latent_dim,
            draw_multiplier = model_info.draw_multiplier;

        // console.log('Generating image...');
        this.model_promise.then((model) => {
            return resolve_after_ms(model, ui_delay_before_tf_computing_ms);
        }).then((model) => {
            return computing_generate_main(model, model_size, draw_multiplier, model_latent_dim, psd);
        });
    }
}
