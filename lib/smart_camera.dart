import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/rendering.dart';
import 'package:image/image.dart' as imglib;
import 'package:camera/camera.dart';
import 'package:device_info/device_info.dart';
import 'package:firebase_ml_vision/firebase_ml_vision.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_widgets/flutter_widgets.dart';
import 'package:path_provider/path_provider.dart';

export 'package:camera/camera.dart';

part 'utils.dart';

typedef HandleDetection<T> = Future<T> Function(FirebaseVisionImage image);
typedef Widget ErrorWidgetBuilder(BuildContext context, CameraError error);

enum CameraError {
  unknown,
  cantInitializeCamera,
  androidVersionNotSupported,
  noCameraAvailable,
}

enum _CameraState { 
  loading,
  error,
  ready
}

class SmartCamera<T> extends StatefulWidget {

  final HandleDetection<T> detector;
  final Function(T, CameraImage) onResult;
  final Function(String) onError;
  final WidgetBuilder loadingBuilder;
  final ErrorWidgetBuilder errorBuilder;
  final WidgetBuilder overlayBuilder;
  final CameraLensDirection cameraLensDirection;
  final ResolutionPreset resolution;
  final Function onDispose;
  final ImageRotation imageRotation;

  const SmartCamera({
    Key key,
    @required this.onResult,
    @required this.onError,
    @required this.detector,
    this.loadingBuilder,
    this.errorBuilder,
    this.overlayBuilder,
    this.cameraLensDirection = CameraLensDirection.back,
    this.resolution = ResolutionPreset.high,
    this.onDispose,
    this.imageRotation = ImageRotation.rotation0
  }) : super(key: key);

  static SmartCamera<List<Face>> faceDetection({
    final FaceDetectorOptions faceDetectorOptions = const FaceDetectorOptions(),
    final Function(List<Face>, CameraImage) onResult,
    final Function(String) onError,
    final WidgetBuilder loadingBuilder,
    final ErrorWidgetBuilder errorBuilder,
    final WidgetBuilder overlayBuilder,
    final CameraLensDirection cameraLensDirection,
    final ResolutionPreset resolution,
    final Function onDispose,
    final bool horizontal,
    final ImageRotation imageRotation = ImageRotation.rotation0
  }) => SmartCamera<List<Face>>(
    detector: FirebaseVision.instance.faceDetector(
      faceDetectorOptions
    ).processImage,
    onResult: onResult,
    onError: onError,
    loadingBuilder: loadingBuilder,
    errorBuilder: errorBuilder,
    overlayBuilder: overlayBuilder,
    cameraLensDirection: cameraLensDirection,
    resolution: resolution,
    onDispose: onDispose,
    imageRotation: imageRotation,
  );  

  static SmartCamera<VisionText> textRecognizer({
    final Function(VisionText, CameraImage) onResult,
    final Function(String) onError,
    final WidgetBuilder loadingBuilder,
    final ErrorWidgetBuilder errorBuilder,
    final WidgetBuilder overlayBuilder,
    final CameraLensDirection cameraLensDirection,
    final ResolutionPreset resolution,
    final Function onDispose,
    final bool horizontal,
    final ImageRotation imageRotation = ImageRotation.rotation0
  }) => SmartCamera<VisionText>(
    detector: FirebaseVision.instance.textRecognizer().processImage,
    onResult: onResult,
    onError: onError,
    loadingBuilder: loadingBuilder,
    errorBuilder: errorBuilder,
    overlayBuilder: overlayBuilder,
    cameraLensDirection: cameraLensDirection,
    resolution: resolution,
    onDispose: onDispose,
    imageRotation: imageRotation,
  );

  /// This method convert an CameraImage object to File
  /// [image] Is the cameraImage returned from onResult method
  /// If your picture is horizontal, then send the second argument false
  static Future<File> convertCameraImagetoFile(CameraImage image, [bool vertical = true]) async {

    final String path = await getPath();

    try {

      final int width = image.width;
      final int height = image.height;
      final int uvRowStride = image.planes[1].bytesPerRow;
      final int uvPixelStride = image.planes[1].bytesPerPixel;

      imglib.Image img = imglib.Image(width, height);

      for(int x=0; x < width; x++) {

        for(int y=0; y < height; y++) {

          final int uvIndex = uvPixelStride * (x/2).floor() + uvRowStride*(y/2).floor();
          final int index = y * width + x;

          final int yp = image.planes[0].bytes[index];
          final int up = image.planes[1].bytes[uvIndex];
          final int vp = image.planes[2].bytes[uvIndex];

          final int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
          final int g = (yp - up * 46549 / 131072 + 44 -vp * 93604 / 131072 + 91).round().clamp(0, 255);
          final int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);

          img.data[index] = shift | (b << 16) | (g << 8) | r;
        }
      }

      if(vertical) img = imglib.copyRotate(img, 90);

      imglib.PngEncoder pngEncoder = imglib.PngEncoder(level: 0, filter: 0);
      final List<int> png = pngEncoder.encodeImage(img);

      final File file = await File('$path/${DateTime.now().toIso8601String()}.png').writeAsBytes(png);

      return file;
    } catch (e) {
      print(">>>>>>>>>>>> ERROR:" + e.toString());
    }
    return null;
  }

  /// This method convert an CameraImage object to base64 image
  /// [image] Is the cameraImage returned from onResult method
  static Future<String> convertCameraImagetoBase64(CameraImage image) async {

    try {

      final int width = image.width;
      final int height = image.height;
      final int uvRowStride = image.planes[1].bytesPerRow;
      final int uvPixelStride = image.planes[1].bytesPerPixel;

      imglib.Image img = imglib.Image(width, height);

      for(int x=0; x < width; x++) {

        for(int y=0; y < height; y++) {

          final int uvIndex = uvPixelStride * (x/2).floor() + uvRowStride*(y/2).floor();
          final int index = y * width + x;

          final int yp = image.planes[0].bytes[index];
          final int up = image.planes[1].bytes[uvIndex];
          final int vp = image.planes[2].bytes[uvIndex];

          final int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
          final int g = (yp - up * 46549 / 131072 + 44 -vp * 93604 / 131072 + 91).round().clamp(0, 255);
          final int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);

          img.data[index] = shift | (b << 16) | (g << 8) | r;
        }
      }

      img = imglib.copyRotate(img, 90);

      final Uint8List bytes = img.getBytes();
      final String base64Image = base64Encode(bytes);

      return base64Image;
    } catch (e) {
      print(">>>>>>>>>>>> ERROR:" + e.toString());
    }
    return null;
  }

  /// [imageFile] Is the File returned from convertCameraImagetoFile method
  /// [face] Is the last face returned from onResult method
  static Future<File> getFaceFromFile(File imageFile, Face face) async {

    final String path = await getPath();
    
    final imglib.Image image = _copyCrop(
      src: imglib.decodeImage(imageFile.readAsBytesSync()),
      x: face.boundingBox.topLeft.dx.toInt(),
      y: face.boundingBox.topLeft.dy.toInt() - 100,
      h: face.boundingBox.height.toInt() + 110,
      w: face.boundingBox.width.toInt(),
    );

    return File('$path/${Random().nextInt(1000000)}.png')..writeAsBytesSync(imglib.encodePng(image));
  }

  static imglib.Image _copyCrop({
    imglib.Image src,
    int x,
    int y,
    int w,
    int h
  }) {

    x = x.clamp(0, src.width - 1).toInt();
    y = y.clamp(0, src.height - 1).toInt();

    if (x + w > src.width) {
      w = src.width - x;
    }
    if (y + h > src.height) {
      h = src.height - y;
    }

    final imglib.Image dst = imglib.Image(
      w,
      h,
      channels: src.channels,
      exif: src.exif,
      iccp: src.iccProfile
    );

    for (var yi = 0, sy = y; yi < h; ++yi, ++sy) {
      for (var xi = 0, sx = x; xi < w; ++xi, ++sx) {
        dst.setPixel(xi, yi, src.getPixel(sx, sy));
      }
    }

    return dst;
  }

  static Uint8List concatenatePlanes(List<Plane> planes) {
    final WriteBuffer allBytes = WriteBuffer();
    planes.forEach((plane) => allBytes.putUint8List(plane.bytes));
    return allBytes.done().buffer.asUint8List();
  }

  @override
  _SmartCameraState createState() => _SmartCameraState<T>();

}

class _SmartCameraState<T> extends State<SmartCamera<T>> with WidgetsBindingObserver {
  
  String _lastImage;
  Key _visibilityKey = UniqueKey();
  CameraController _cameraController;
  ImageRotation _rotation;
  _CameraState _smartCameraState = _CameraState.loading;
  CameraError _cameraError = CameraError.unknown;
  bool _alreadyCheckingImage = false;
  bool _isStreaming = false;
  bool _isDeactivate = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initialize();
  }

  @override
  void didUpdateWidget(SmartCamera<T> oldWidget) {
    if (oldWidget.resolution != widget.resolution) {
      _initialize();
    }
    super.didUpdateWidget(oldWidget);
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // App state changed before we got the chance to initialize.
    if (_cameraController == null || !_cameraController.value.isInitialized) {
      return;
    }
    if (state == AppLifecycleState.inactive) {
      _cameraController?.dispose();
    } else if (state == AppLifecycleState.resumed && _isStreaming) {
      _initialize();
    }
  }

  Future<void> stop() async {
    if (_cameraController != null) {
      if (_lastImage != null && File(_lastImage).existsSync()) {
        // this.widget.file = File(_lastImage);
        // await File(_lastImage).delete();
      }

      final Directory tempDir = await getTemporaryDirectory();
      _lastImage = '${tempDir.path}/${DateTime.now().millisecondsSinceEpoch}';

      try {
        await _cameraController.initialize();
        await _cameraController.takePicture();
      } on PlatformException catch (e) {
        debugPrint('$e');
      }

      await _stop(false);
    }
  }

  Future<void> _stop(bool silently) {
    final completer = Completer();
    scheduleMicrotask(() async {
      if (_cameraController?.value?.isStreamingImages == true && mounted) {
        await _cameraController.stopImageStream().catchError((_) {});
      }

      if (silently) {
        _isStreaming = false;
      } else {
        setState(() {
          _isStreaming = false;
        });
      }
      completer.complete();
    });
    return completer.future;
  }

  void start() {
    if (_cameraController != null) {
      _start();
    }
  }

  void _start() {
    _cameraController.startImageStream(_processImage);
    setState(() {
      _isStreaming = true;
    });
  }

  CameraValue get cameraValue => _cameraController?.value;
  ImageRotation get imageRotation => _rotation;

  Future<void> Function() get prepareForVideoRecording =>
      _cameraController.prepareForVideoRecording;

  Future<void> startVideoRecording(String path) async {
    await _cameraController.stopImageStream();
    return _cameraController.startVideoRecording();
  }

  Future<void> stopVideoRecording() async {
    await _cameraController.stopVideoRecording();
    await _cameraController.startImageStream(_processImage);
  }

  CameraController get cameraController => _cameraController;

  Future<void> takePicture(String path) async {
    await _stop(false);
    await _cameraController.initialize();
    await _cameraController.takePicture();
    _start();
  }

  Future<void> _initialize() async {
    if (Platform.isAndroid) {
      final deviceInfo = DeviceInfoPlugin();
      final androidInfo = await deviceInfo.androidInfo;
      if (androidInfo.version.sdkInt < 21) {
        debugPrint('Camera plugin doesn\'t support android under version 21');
        if (mounted) {
          setState(() {
            _smartCameraState = _CameraState.error;
            _cameraError = CameraError.androidVersionNotSupported;
          });
        }
        return;
      }
    }

    final CameraDescription description = await _getCamera(
      widget.cameraLensDirection
    );

    if (description == null) {
      _smartCameraState = _CameraState.error;
      _cameraError = CameraError.noCameraAvailable;

      return;
    }
    await _cameraController?.dispose();
    _cameraController = CameraController(
      description,
      widget.resolution ?? ResolutionPreset.high,
      enableAudio: false,
    );
    if (!mounted) {
      return;
    }

    try {
      await _cameraController.initialize();
    } catch (ex, stack) {
      debugPrint('Can\'t initialize camera');
      debugPrint('$ex, $stack');
      if (mounted) {
        setState(() {
          _smartCameraState = _CameraState.error;
          _cameraError = CameraError.cantInitializeCamera;
        });
      }
      return;
    }

    if (!mounted) {
      return;
    }

    setState(() {
      _smartCameraState = _CameraState.ready;
    });
    _rotation = _rotationIntToImageRotation(
      description.sensorOrientation,
    );

    //FIXME hacky technique to avoid having black screen on some android devices
    await Future.delayed(Duration(milliseconds: 200));
    start();
  }

  @override
  void dispose() {
    if (widget.onDispose != null) {
      widget.onDispose();
    }
    // if (_lastImage != null && File(_lastImage).existsSync()) {
    //   this.file = File(_lastImage);
    //   File(_lastImage).delete();
    // }

    if (_cameraController != null) {
      _cameraController.dispose();
    }
    _cameraController = null;
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_smartCameraState == _CameraState.loading) {
      return widget.loadingBuilder == null
          ? Center(child: CircularProgressIndicator())
          : widget.loadingBuilder(context);
    }
    if (_smartCameraState == _CameraState.error) {
      return widget.errorBuilder == null
          ? Center(child: Text('$_smartCameraState $_cameraError'))
          : widget.errorBuilder(context, _cameraError);
    }

    Widget cameraPreview = AspectRatio(
      aspectRatio: _cameraController.value.isInitialized ? _cameraController.value.aspectRatio : 1,
      child: _isStreaming
          ? CameraPreview(
        _cameraController,
      )
          : _getPicture(),
    );

    if (widget.overlayBuilder != null) {
      cameraPreview = Stack(
        fit: StackFit.passthrough,
        children: [
          cameraPreview,
          widget.overlayBuilder(context),
        ],
      );
    }
    return VisibilityDetector(
      child: FittedBox(
        alignment: Alignment.center,
        fit: BoxFit.cover,
        child: SizedBox(
          width: _cameraController.value.previewSize.height *
              _cameraController.value.aspectRatio,
          height: _cameraController.value.previewSize.height,
          child: cameraPreview,
        ),
      ),
      onVisibilityChanged: (VisibilityInfo info) {
        if (info.visibleFraction == 0) {
          //invisible stop the streaming
          _isDeactivate = true;
          _stop(true);
        } else if (_isDeactivate) {
          //visible restart streaming if needed
          _isDeactivate = false;
          _start();
        }
      },
      key: _visibilityKey,
    );
  }

  void _processImage(CameraImage cameraImage) async {

    if (!_alreadyCheckingImage && mounted) {
      _alreadyCheckingImage = true;
      try {
        final T results = await _detect<T>(
          cameraImage,
          widget.detector,
          widget.imageRotation
        );
        widget.onResult(results, cameraImage);
      } catch (err, _) {
        this.widget.onError('$err');
      }
      _alreadyCheckingImage = false;
    }
  }

  void toggle() {
    if (_isStreaming && _cameraController.value.isStreamingImages) {
      stop();
    } else {
      start();
    }
  }

  Widget _getPicture() {
    if (_lastImage != null) {
      final file = File(_lastImage);
      // this.file = file;
      if (file.existsSync()) {
        return Image.file(file);
      }
    }

    return Container();
  }
}

