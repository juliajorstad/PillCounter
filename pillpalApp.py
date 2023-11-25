from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFillRoundFlatButton
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivymd.uix.button import MDFlatButton, MDRaisedButton
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserIconView
import cv2
import os

from pillDetectionYOLO import predict_with_yolo


# create pop up window for uploading image
class UploadPopup(Popup):
    def __init__(self, on_selection, **kwargs):
        super(UploadPopup, self).__init__(**kwargs)
        self.on_selection = on_selection
        # chose path to folder to upload from
        self.filechooser = FileChooserIconView(path="datasets/test")
        self.content = self.filechooser
        self.filechooser.bind(on_submit=self.on_file_select)

    def on_file_select(self, filechooser, selection, touch):
        if selection:
            self.dismiss()
            self.on_selection(selection[0])


# create pop up window for camera
class CameraPopup(Popup):
    def __init__(self, **kwargs):
        super(CameraPopup, self).__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        self.stream_event = None

        self.layout = BoxLayout(orientation='vertical')
        self.img1 = Image()
        self.layout.add_widget(self.img1)

        # Button to capture or retake image
        self.capture_button = MDRaisedButton(text='Capture', size_hint=(1, 0.1))
        self.capture_button.bind(on_press=self.capture_or_retake_image)
        self.layout.add_widget(self.capture_button)

        # Return button
        return_button = MDFlatButton(text='Return', size_hint=(1, 0.1))
        return_button.bind(on_press=self.close_popup)
        self.layout.add_widget(return_button)

        self.content = self.layout
        self.bind(on_open=self.start_stream)
        self.bind(on_dismiss=self.stop_stream)

    def capture_or_retake_image(self, instance):
        if self.capture_button.text == 'Capture':
            # Capturing the image
            ret, frame = self.capture.read()
            if ret:
                # Pass the captured image to function in pillDetectionYOLO.py module for processing
                processed_frame = predict_with_yolo(frame)

                # Update the Kivy Image widget with the processed frame
                self.update_image_with_frame(processed_frame)

                # stop webcam application
                self.stop_stream()

                # Change button text to "Retake"
                self.capture_button.text = 'Retake'
        else:
            # Retaking the image: restart the capture and stream, change button text back to "Capture"
            self.restart_stream()
            self.capture_button.text = 'Capture'

    def restart_stream(self):
        # Release the existing capture if it exists
        if self.capture is not None:
            self.capture.release()

        # Restart the capture and stream
        self.capture = cv2.VideoCapture(0)
        self.start_stream()

    def start_stream(self, *args):
        if self.stream_event is None or not self.stream_event.is_triggered:
            self.stream_event = Clock.schedule_interval(self.update_image, 1.0 / 30.0)

    def stop_stream(self, *args):
        if self.stream_event is not None and self.stream_event.is_triggered:
            Clock.unschedule(self.stream_event)
            if self.capture is not None:
                self.capture.release()

    def close_popup(self, instance):
        self.stop_stream()
        self.dismiss()

    def update_image(self, *args):
        ret, frame = self.capture.read()
        if ret:
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img1.texture = texture

    def update_image_with_frame(self, frame):
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img1.texture = texture

    def capture_image(self, instance):
        ret, frame = self.capture.read()
        if ret:
            # Process the captured frame with YOLO
            processed_frame = predict_with_yolo(frame)

            # Update the Kivy Image widget with the processed frame
            self.update_image_with_frame(processed_frame)

            # Optionally, stop the live stream
            self.stop_stream()


# pop up window for main menu
class PillPalApp(MDApp):
    def build(self):
        # App title
        self.theme_cls.primary_palette = "Green"
        layout = MDBoxLayout(orientation='vertical', padding=10, spacing=10)

        """title_label = MDLabel(text="PillPal", halign="center",font_style="H1")
        layout.add_widget(title_label)"""

        # Image display

        self.display_img = Image(source='kivy_app/App Logo Ideas- 3.jpg', size_hint=(None, None), size=(700, 700),
                                 pos_hint={'center_x': 0.5})
        layout.add_widget(self.display_img)

        spacer = Widget(size_hint_y=None, height=200)  # Adjust height as needed
        layout.add_widget(spacer)
        # Start button
        start_button = MDFillRoundFlatButton(text="Start Camera", pos_hint={"center_x": 0.5})
        start_button.bind(on_press=self.on_start_button_press)
        layout.add_widget(start_button)

        # Upload button
        upload_button = MDFlatButton(text="Upload Image", pos_hint={"center_x": 0.5})
        upload_button.bind(on_press=self.on_upload_button_press)
        layout.add_widget(upload_button)

        # Quit button
        quit_button = MDFlatButton(text="Quit", pos_hint={"center_x": 0.5})
        quit_button.bind(on_press=self.stop_app)
        layout.add_widget(quit_button)

        return layout

    def on_start_button_press(self, instance):
        popup = CameraPopup(title='Camera Stream', size_hint=(0.9, 0.9))
        popup.open()

    def on_upload_button_press(self, instance):
        popup = UploadPopup(on_selection=self.process_uploaded_image)
        popup.open()

    def process_uploaded_image(self, file_path):
        if os.path.exists(file_path):
            # Process the uploaded image
            img = cv2.imread(file_path)
            processed_frame = predict_with_yolo(img)
            self.update_image_with_frame(processed_frame)

    def update_image_with_frame(self, frame):
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.display_img.texture = texture

    def stop_app(self, instance):
        self.stop()


if __name__ == '__main__':
    PillPalApp().run()
