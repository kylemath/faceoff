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

let dampingOfChange = 50; //smaller is more change

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
        
        let c = document.getElementById("the_canvas");
        await tf.browser.toPixels(outPixels, c);
        await tf.nextFrame();
    };
}

// async function computing_animate_latent_space(model, draw_multiplier, animate_frame) {
//     const inputShape = model.inputs[0].shape.slice(1);
//     const shift = tf.randomNormal(inputShape).expandDims(0);
//     const freq = tf.randomNormal(inputShape, 0, .1).expandDims(0);

//     let c = document.getElementById("the_canvas");
//     let i = 0;
//     while (i < animate_frame) {
//         i++;
//         const y = tf.tidy(() => {
//             const z = tf.sin(tf.scalar(i).mul(freq).add(shift));
//             const y = model.predict(z).squeeze().transpose([1, 2, 0]).div(tf.scalar(2)).add(tf.scalar(.5));
//             return image_enlarge(y, draw_multiplier);
//         });

//         await tf.toPixels(y, c);
//         await tf.nextFrame();
//     }
// }

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
