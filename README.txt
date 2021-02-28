Directory progetto

    detection.py
        carica pesi
        nomi classi
        prepare img
        usa modello
        drawboxes

    yolov3.cfg
        BATCH_NORM_DECAY
        BATCH_NORM_EPSILON
        LEAKY_RELU
        ANCHORS
        WEIGHTS_PATH
        NUM_CLASS
        LABELS_PATH
        MODEL_IMG_SIZE
        IOU_THRESHOLD               (precisione delle bb)
        CONFIDENCE_THRESHOLD        (prob che ci sia un oggetto)

    yolov3.weights 

    [LAYER]
        common_layers.py
            batch_norm              [KEGGLE MA NEL CASO USA TF]
            fixed_padding
            conv 2d padding         [MIX KEG & DATAH, SEQUENZIALE]
            darknet_residual        [KEGGLE]
            yolo_conv_block         [KEGGLE CON F NOSTRE]
            yolo_layer              [DI BASE KEGGLE MA VEDI]
        darknet53.py (Feature Extraction)    
            vedi disegno            [KEGGLE CON F NOSTRE]
        yolov3_model.py
            upsample                [DATAH MA ASPETTO CHECK]
            modello                 [DA CONTROLLARE PER BENE]
    
    [DATASET] (ref COCO)
        labels.txt
        test.jpg
    
    [UTILS]
        Parser CFG.py
        dataset.py
        load_weights.py (as given)
        boxes.py
            draw_boxes              [DATAH SE AVANZA TEMPO KEG]
            build_boxes             [KEGGLE]
            nms                     [DI BASE DATAH MA VEDI OUTPUT]