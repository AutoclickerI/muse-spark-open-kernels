# Lab b964cb5d180a463b: two leading expert choices per T. Keep the old
# measured incumbent eligible where its ranking disagreed with the warm lab.
_SWEEP_EXPERT_DEFAULTS = {753:'raw17',3012:'humming_masked',16384:'dg_w8'}
_SWEEP_EXPERT_PRIOR = {753:'humming_masked',3012:'humming_masked',16384:'dg_hint_w8'}
_SWEEP_EXPERT_CHOICES = {
    753:('raw17','tma_cluster','humming_masked'),
    3012:('humming_masked','humming_mid'),
    16384:('dg_w8','humming_masked','dg_hint_w8'),
}


class _SweepExpertGraph(_PgGraph):
    def replay(self):
        self.graph.replay()

    @property
    def output(self):
        return self.out


def _sweep_expert_verify(graph,fixtures,static,refs,indices,where):
    worst={'relative_l2':0.,'worst_row':0.}
    for fixture,ref in zip(fixtures,refs):
        for dst,src in zip(static,fixture):dst.copy_(src)
        graph.replay()
        error=_judge_floor(graph.output[indices],ref,where)
        for key in worst:worst[key]=max(worst[key],error[key])
    return worst


def _tune_score_experts(st,flush):
    """Fresh realistic routes, FP32 admission, then20 cold alternating pairs.

    The warm eager sweep supplies defaults/shortlists, never final timing.
    Both leading sweep choices are always captured; an old winner can compete
    too. No uniform sweep consumes the prepare budget before these pairs.
    """
    st['_expert_sweep_pairs']={}
    for T,choices in _SWEEP_EXPERT_CHOICES.items():
        if _time_left(st)<12:
            print('SCORE_TUNING_BUDGET expert sweep pairs omitted',T,flush=True)
            return
        graphs=[];names=[];errors={};route_stats=[]
        prior=_SWEEP_EXPERT_PRIOR[T]
        record=dict(choices=[],us=[],selected=st['expert_dispatch'][T],errors={})
        try:
            fixtures=[_expert_realistic(T,st) for _ in range(3)]
            static=tuple(value.clone() for value in fixtures[0])
            indices=torch.linspace(0,T-1,min(64,T),device=static[0].device).round().long()
            refs=[]
            for fixture in fixtures:
                sampled=tuple(value[indices] if i<4 else value for i,value in enumerate(fixture))
                refs.append(_score_expert_oracle(sampled,st))
                counts=fixture[-1]
                route_stats.append(dict(total=int(counts.sum().item()),
                    distinct=int((counts>0).sum().item()),hot=int(counts.max().item())))
            # If all new contenders reject, retain the prior whole-layer winner.
            st['expert_dispatch'][T]=prior
            for name in choices:
                graph=None;accepted=False
                try:
                    if _time_left(st)<5:raise TimeoutError('expert graph admission budget')
                    graph=_SweepExpertGraph(lambda name=name:_score_expert(*static,st,name))
                    errors[name]=_sweep_expert_verify(graph,fixtures,static,refs,indices,
                                                     f'expert sweep {T} {name}')
                    graphs.append(graph);names.append(name);accepted=True
                except Exception as error:
                    _score_reject(f'expert sweep admission {T} {name}',error)
                finally:
                    if graph is not None and not accepted:graph.close()
            if not graphs:raise RuntimeError('no admitted expert graph; retain previous whole-layer winner')
            control_name=_SWEEP_EXPERT_DEFAULTS[T]
            control=names.index(control_name) if control_name in names else 0
            def refresh(rep):
                if _time_left(st)<3:raise TimeoutError('expert paired timing budget')
                # The two order-swapped repeats share the same freshly copied
                # pooled values, IDs, probabilities and per-expert counts.
                for dst,src in zip(static,fixtures[(rep//2)%len(fixtures)]):dst.copy_(src)
            costs=_score_times(graphs,refresh,flush,repeats=20,control=control)
            live=[]
            for index,(name,graph) in enumerate(zip(names,graphs)):
                try:
                    if not math.isfinite(costs[index]):raise RuntimeError('paired timing rejected')
                    errors[name]=_sweep_expert_verify(graph,fixtures,static,refs,indices,
                                                     f'expert sweep final {T} {name}')
                    live.append(index)
                except Exception as error:
                    _score_reject(f'expert sweep final {T} {name}',error)
            if not live:raise RuntimeError('all paired finalists rejected; retain previous whole-layer winner')
            winner=min(live,key=lambda index:costs[index])
            selected=names[winner]
            st['expert_dispatch'][T]=selected
            record=dict(choices=[names[index] for index in live],us=[costs[index] for index in live],
                        selected=selected,errors={names[index]:errors[names[index]] for index in live})
            st['_expert_sweep_pairs'][T]=dict(default=_SWEEP_EXPERT_DEFAULTS[T],prior=prior,
                selected=selected,choices=record['choices'],paired_us=record['us'],
                errors=record['errors'],repeats=20,fixtures=route_stats)
            print('SCORE_SELECT',json.dumps(dict(stage=f'expert sweep paired {T}',
                selected=selected,finalists=record['choices'],paired_us=record['us'],
                errors=record['errors'],fixtures=route_stats,repeats=20)),flush=True)
        except Exception as error:
            _score_reject(f'expert sweep shape {T}',error)
            record['selected']=st['expert_dispatch'][T]
        finally:
            for graph in graphs:graph.close()
        st['_measurements']['expert'][T]=record


def _expert_sweep_last(st):
    # Print after dense/group details, immediately before SCORE_LAST_SUMMARY.
    print('EXPERT_SWEEP_LAST',json.dumps(dict(source='b964cb5d180a463b',
        defaults=_SWEEP_EXPERT_DEFAULTS,selected=st['expert_dispatch'],
        pairs=st.get('_expert_sweep_pairs',{})),separators=(',',':')),flush=True)
