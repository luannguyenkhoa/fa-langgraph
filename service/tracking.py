import os
from amplitude import Amplitude, BaseEvent

def callback_fun(e, code, message):
    """A callback function"""
    print(e)
    print(code, message)


# Initialize a Amplitude client instance
amp_client = Amplitude(api_key=os.environ.get("AMPLITUDE_API_KEY"))
# Config a callback function
amp_client.configuration.callback = callback_fun

def track_event(name, session, properties):
    amp_client.track(
        BaseEvent(
            event_type=name,
            device_id=session,
            event_properties=properties
        )
    )
