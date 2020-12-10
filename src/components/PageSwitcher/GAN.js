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

function computing_prep_canvas(size) {
    // We don't `return tf.image.resizeBilinear(v1, [size * draw_multiplier, size * draw_multiplier]);`
    // since that makes image blurred, which is not what we want.
    // So instead, we manually enlarge the image.
    let canvas = document.getElementById("the_canvas");
    let ctx = canvas.getContext("2d");
    ctx.canvas.width = size;
    ctx.canvas.height = size;
}

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

async function computing_generate_main(model, size, draw_multiplier, latent_dim) {
    const y = tf.tidy(() => {
        const z = tf.randomNormal([1, latent_dim]);
        console.log(z)
        const y = model.predict(z).squeeze().transpose([1, 2, 0]).div(tf.scalar(2)).add(tf.scalar(0.5));
        console.log(y)
        return image_enlarge(y, draw_multiplier);

    });
    let c = document.getElementById("the_canvas");
    console.log('canvas', c);
    console.log('image generated, image data: ', y);
    await tf.browser.toPixels(y, c);
}

const ui_delay_before_tf_computing_ms = 20;  // Delay that many ms before tf computing, which can block UI drawing.

function resolve_after_ms(x, ms) {
    return new Promise(resolve => {
        setTimeout(() => {
            resolve(x);
        }, ms);
    });
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
        let model_size = model_info.model_size,
            model_url = model_info.model_url,
            draw_multiplier = model_info.draw_multiplier,
            description = model_info.description;

        computing_prep_canvas(model_size * draw_multiplier);
        console.log(`Setting up model ${description}`);

        // ui_set_canvas_wrapper_size(model_size * draw_multiplier);
        // ui_logging_set_text(`Setting up model ${description}...`);

        if (model_name in this.model_promise_cache) {
            this.model_promise = this.model_promise_cache[model_name];
            // ui_logging_set_text(`Model "${description}" is ready.`);
        } else {
            // ui_generate_button_disable('Loading...');
            // ui_animate_button_disable('Loading...');
            // ui_logging_set_text(`Loading model "${description}"...`);
            this.model_promise = tf.loadLayersModel(model_url);
            this.model_promise.then((model) => {
                return resolve_after_ms(model, ui_delay_before_tf_computing_ms);
            }).then((model) => {
                // ui_generate_button_enable();
                // ui_animate_button_enable();
                // ui_logging_set_text(`Model "${description}" is ready.`);
                console.log(`Model "${description} is ready.`);
            });
            this.model_promise_cache[model_name] = this.model_promise;
        }
    }

    generate() {
        let model_info = all_model_info[this.model_name];
        let model_size = model_info.model_size,
            model_latent_dim = model_info.model_latent_dim,
            draw_multiplier = model_info.draw_multiplier;

        computing_prep_canvas(model_size * draw_multiplier);

        console.log('Generating image...');

        // ui_logging_set_text('Generating image...');
        // ui_generate_button_disable('Generating...');
        // ui_animate_button_disable();

        this.model_promise.then((model) => {
            return resolve_after_ms(model, ui_delay_before_tf_computing_ms);
        }).then((model) => {
            this.start_ms = (new Date()).getTime();
            return computing_generate_main(model, model_size, draw_multiplier, model_latent_dim);
        }).then((_) => {
            let end_ms = (new Date()).getTime();
            // ui_generate_button_enable();
            // ui_animate_button_enable();
            // ui_logging_set_text(`Image generated. It took ${(end_ms - this.start_ms)} ms.`);
        });
    }
}
