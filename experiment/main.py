from datetime import datetime
import asyncio
import random
import reactivex as rx
from recording import Recording
from mindwave import Headset
from pathlib import Path

# Parameters
WORDS = ['RIGHT', 'LEFT']
SETS = 15
TRIAL_NAME = datetime.now().strftime("%Y-%m-%d-%H%M%S")

async def main():
    headset = Headset()
    stimuli = rx.Subject()
    recording = Recording(headset, stimuli)

    try:
        await headset.connect('/dev/tty.MindWaveMobile')
        print('Ready!')
        
        order = WORDS * SETS
        random.shuffle(order)

        recording.start()

        for word in order:
            spoken = random.random() > .5
            prompt = 'SAY' if spoken else 'THINK'
            print(f'\nOn the count of three, please {prompt} the word: {word}')
            for i in range(3, 0, -1):
                await asyncio.sleep(2)
                print(i)
            
            await asyncio.sleep(1)
            recording.start()
            await asyncio.sleep(1)
            
            print('Go!')
            stimuli.on_next(f'{prompt}-{word}')
            await asyncio.sleep(5)

            print('Stop!')
            await asyncio.sleep(1)
            recording.pause()

        data_dir = Path('.', 'data', TRIAL_NAME)
        data_dir.mkdir(parents=True, exist_ok=True)
        recording.save(data_dir)
    finally:
        print('Shutting Down')
        headset.disconnect()

if __name__ == '__main__':
    asyncio.run(main())