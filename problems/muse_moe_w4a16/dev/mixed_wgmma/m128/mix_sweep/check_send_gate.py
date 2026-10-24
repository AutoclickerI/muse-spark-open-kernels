"""Check the scheduled dispatcher without sleeping or contacting the service."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import types

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[5]


def main():
    spec=importlib.util.spec_from_file_location('send_gate_check',ROOT/'tools/okc_run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    with tempfile.TemporaryDirectory() as folder:
        m.WAITING=Path(folder)/'waiting';m.LOCK=Path(folder)/'lock';m.LOCK.touch();clock=[1000.];events=[]
        def sleep(seconds):events.append(('sleep',seconds));clock[0]+=seconds
        m.time=types.SimpleNamespace(time=lambda:clock[0],sleep=sleep)
        m.signal=types.SimpleNamespace(pthread_sigmask=lambda *args:None,SIG_BLOCK=0,SIGUSR1=1)
        def wake():
            events.append(('wake',clock[0],len(m.waiters())))
            if clock[0]<1100.:assert len(m.waiters())==0,'scheduled job must not block ready jobs'
        m.wake_waiters=wake
        def live():events.append(('service_read',clock[0]));assert clock[0]>=1100.;return []
        m.in_flight=live
        def mode():assert clock[0]>=1100.;return 'check'
        def send(mode):events.append(('send',clock[0]));assert clock[0]>=1100.;return mode,'local_fake'
        result=m.send_in_turn(dict(problem='local_fake',not_before=1100.),1,send,mode)
        assert result==('check','local_fake') and not list(m.WAITING.glob('*.json'))
    assert m.parse_not_before('2026-10-25T09:01:00+09:00')==m.parse_not_before('2026-10-25T00:01:00Z')
    try:m.parse_not_before('2026-10-25T09:01:00')
    except argparse.ArgumentTypeError:pass
    else:raise AssertionError('ambiguous timezone accepted')
    report=dict(no_real_service_calls=True,no_real_sleep=True,held_outside_queue_order=True,no_service_read_or_send_before_gate=True,events=events,timezone_equivalence=True,ambiguous_timestamp_rejected=True)
    (HERE/'send_gate.json').write_text(json.dumps(report,indent=2)+'\n');print('SEND_GATE_PASS',events)


if __name__=='__main__':main()
